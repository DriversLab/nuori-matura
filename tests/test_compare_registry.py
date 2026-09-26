"""Compare math, paired statistics, the promotion gate and the run registry.

All fixtures are synthetic files under tmp_path: no models, no network, never the real results/ or checkpoints/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from eval.compare import compare_runs, format_compare_table, write_compare
from eval.grader import summarize
from eval.schema import write_jsonl
from eval.stats import paired_bootstrap_ci, sign_test
from train import registry
from train.config import ROOT, checkpoint_dir, config_hash, dump_config, load_config, results_dir, run_dir

GATE = {
    "min_delta_points": 0.5,
    "max_parse_fail_rate_increase": 0.0,
    "max_general_nll_increase": 0.05,
    "max_variant_regression_points": 1.0,
    "max_secondary_score_drop_pct": 2.0,
    "min_net_score_pct": 0.0,
    "max_loglik_acc_drop_pct": 3.0,
    "must_beat_current_candidate": True,
    "min_improvement_points": 0.0,
    "max_lenient_regression_points": 0.0,
}
EXT = "ext_llmzszl_matura"  # set names mirror configs/base.yaml eval.sets, the reference of gate [scope]
FP_PARTS = {
    "format_version": "v1",
    "sets": [
        {"name": "heldout", "sha256": "a" * 64, "variants": ["canonical", "bare", "payload_only"], "loglik": True},
        {"name": EXT, "sha256": "b" * 64, "variants": ["canonical"], "loglik": True},
    ],
    "max_new_tokens": 48,
    "system_prompt": None,
    "limit": None,
    "general_loss": {"sha256": "c" * 64, "max_items": 300, "max_length": 1024},
    "dtype": "bfloat16",
    "quantization": "none",
    "device": "cuda",
}
MERGED_FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
KEPT_FILES = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "train_metrics.json",
              "resolved_config.yaml", "training_meta.json")
# the organizers' prawko-v2 eval splits are gitignored (third-party rows): tests that need them skip on a fresh clone
PRAWKO_EVAL_FILES = tuple(ROOT / "data" / "eval" / f"ext_prawko_{split}.jsonl" for split in ("dev", "test"))
needs_prawko_files = pytest.mark.skipif(not all(p.exists() for p in PRAWKO_EVAL_FILES),
                                        reason="data/eval/ext_prawko_*.jsonl missing (gitignored): "
                                               "re-create with python scripts/import_prawko.py")


# ----------------------------------------------------------------------------- synthetic eval fixtures


def rec(item_id: str, subject: str, points: float, max_points: float = 1.0, *, parse_ok: bool = True,
        lenient: float | None = None) -> dict:
    """A grader record (the fields eval.grader.summarize and eval.compare read)."""
    return {
        "id": item_id, "subject": subject, "type": "mc", "max_points": float(max_points), "points": float(points),
        "correct": points == max_points, "parse_ok": parse_ok, "parse_reason": None if parse_ok else "no_answer_line",
        "points_lenient": float(points if lenient is None else lenient),
    }


HELDOUT = [("h1", "historia", 1), ("h2", "historia", 2), ("h3", "biologia", 1), ("h4", "biologia", 3),
           ("h5", "matematyka", 1), ("h6", "matematyka", 1)]


def heldout(points: list[float], *, parse_fail: tuple[str, ...] = (), lenient: dict[str, float] | None = None) -> list[dict]:
    return [rec(i, s, p, m, parse_ok=i not in parse_fail, lenient=(lenient or {}).get(i)) for (i, s, m), p in zip(HELDOUT, points)]


def ext(points: list[float]) -> list[dict]:
    return [rec(f"e{k}", "fizyka", p) for k, p in enumerate(points, 1)]


# base: canonical 3/9 (h5 unparseable, h6 lenient-only), bare 3/9, payload_only 3/9, ext 2/4
BASE_RECORDS = {
    "heldout": {"canonical": heldout([1, 0, 1, 1, 0, 0], parse_fail=("h5",), lenient={"h6": 1}), "bare": heldout([1, 0, 1, 1, 0, 0]),
                "payload_only": heldout([1, 0, 1, 1, 0, 0])},
    EXT: {"canonical": ext([1, 1, 0, 0])},
}
# run: canonical 7/9 (gains h2 +2, h4 +2, h5 +1; loses h3 -1), bare 2/9 (-1), payload_only 3/9 (0), ext 3/4
RUN_RECORDS = {
    "heldout": {"canonical": heldout([1, 2, 0, 3, 1, 0]), "bare": heldout([0, 0, 1, 1, 0, 0]), "payload_only": heldout([1, 0, 1, 1, 0, 0])},
    EXT: {"canonical": ext([1, 1, 1, 0])},
}


def write_eval(out: Path, name: str, records: dict, *, loglik: dict[str, float], nll: float | None, model: str,
               fingerprint: str = "fp-1", parts: dict | None = None, predictions: bool = True) -> dict:
    """summary.json (+ predictions.<set>.<variant>.jsonl) shaped like eval.runner.run_eval output."""
    out.mkdir(parents=True, exist_ok=True)
    sets: dict[str, dict] = {}
    for set_name, variants in records.items():
        ll = {"n": 4, "acc": loglik[set_name], "per_subject": {}} if set_name in loglik else None
        sets[set_name] = {"path": f"data/eval/{set_name}.jsonl", "n_items": 0, "variants": {}, "loglik": ll}
        for variant, recs in variants.items():
            sets[set_name]["n_items"] = len(recs)
            sets[set_name]["variants"][variant] = summarize(recs)
            if predictions:
                write_jsonl([{**r, "variant": variant} for r in recs], out / f"predictions.{set_name}.{variant}.jsonl")
    head = sets["heldout"]["variants"]["canonical"]
    summary = {
        "run_name": name, "model": model, "is_base": name.startswith("base"), "base_model": "speakleash/Bielik-4.5B-v3.0-Instruct",
        "timestamp": "2026-09-16T10:00:00+00:00", "fingerprint": fingerprint, "fingerprint_parts": parts or FP_PARTS,
        "device": "cuda", "dtype": "bfloat16", "quantization": "none", "primary_set": "heldout", "primary_variant": "canonical",
        "headline": {k: head[k] for k in ("points", "max_points", "score_pct", "parse_fail_rate", "points_lenient")},
        "sets": sets, "general_heldout": None if nll is None else {"nll": nll, "tokens": 1000, "n": 30}, "elapsed_s": 1.0,
    }
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def organizer_block(accuracy: float, *, n: int = 4, abc_mass: float = 0.80, skipped: int = 0,
                    rotated_accuracy: float | None = None) -> dict:
    """summary.json sets.<set>.organizer: eval.organizers.summarize_organizer + "skipped" + "rotated"."""
    def block(acc: float) -> dict:
        correct = round(acc * n)
        return {"n": n, "correct": correct, "accuracy": acc, "mean_correct_probability": acc, "mean_abc_mass": abc_mass,
                "per_subject": {"wos": {"n": n, "correct": correct, "accuracy": acc}}}

    return {**block(accuracy), "skipped": skipped,
            "rotated": None if rotated_accuracy is None else block(rotated_accuracy)}


def add_organizer(out: Path, blocks: dict[str, dict]) -> None:
    """Add organizer blocks to an already written summary.json (the key is additive: older summaries lack it)."""
    path = out / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    for set_name, block in blocks.items():
        summary["sets"][set_name]["organizer"] = block
    path.write_text(json.dumps(summary), encoding="utf-8")


def make_checkpoint(cfg: dict, name: str) -> Path:
    ck = checkpoint_dir(cfg, name)
    for rel in MERGED_FILES + KEPT_FILES:
        (ck / rel).parent.mkdir(parents=True, exist_ok=True)
        (ck / rel).write_text("{}", encoding="utf-8")
    return ck


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    c = load_config(None, {
        "run_name": "run1",
        "tags": [],
        "paths.results_root": str(tmp_path / "results"),
        "paths.checkpoints_root": str(tmp_path / "checkpoints"),
        "data.synthetic": str(tmp_path / "synthetic" / "clean.jsonl"),
        "pipeline.keep_merged": "candidate_only",
    })
    c["gate"] = dict(GATE)
    return c


@pytest.fixture
def eval_dirs(cfg: dict) -> tuple[Path, Path]:
    base, run = run_dir(cfg, "base"), run_dir(cfg, "run1")
    write_eval(base, "base", BASE_RECORDS, loglik={"heldout": 0.5, EXT: 0.40}, nll=1.25, model=cfg["model"]["base"])
    write_eval(run, "run1", RUN_RECORDS, loglik={"heldout": 0.6, EXT: 0.38}, nll=1.30, model=str(checkpoint_dir(cfg, "run1")))
    return base, run


# ----------------------------------------------------------------------------- stats


def test_sign_test_known_p_values() -> None:
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0
    assert sign_test(5, 2) == pytest.approx(0.453125)  # 2 * (1 + 7 + 21) / 2**7
    assert sign_test(0, 5) == pytest.approx(0.0625)
    assert sign_test(1, 9) == pytest.approx(0.021484375)
    assert sign_test(9, 1) == sign_test(1, 9)
    assert sign_test(0, 400) < 1e-100
    with pytest.raises(ValueError):
        sign_test(-1, 2)


def test_sign_test_matches_scipy_binomtest() -> None:
    stats = pytest.importorskip("scipy.stats")
    for wins, losses in [(0, 1), (2, 7), (10, 3), (25, 25), (40, 21)]:
        expected = stats.binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue
        assert sign_test(wins, losses) == pytest.approx(expected, rel=1e-9)


def test_bootstrap_ci_contains_observed_delta_and_is_seeded() -> None:
    import numpy as np

    rng = np.random.default_rng(123)
    base = rng.integers(0, 3, size=60).astype(float).tolist()
    run = [b + d for b, d in zip(base, rng.choice([-1.0, 0.0, 0.0, 1.0, 2.0], size=60))]
    observed = sum(run) - sum(base)
    lo, hi = paired_bootstrap_ci(base, run, n_boot=2000, seed=7)
    assert lo < observed < hi
    assert (lo, hi) == paired_bootstrap_ci(base, run, n_boot=2000, seed=7)
    lo50, hi50 = paired_bootstrap_ci(base, run, n_boot=2000, seed=7, alpha=0.5)
    assert lo <= lo50 < hi50 <= hi


def test_bootstrap_ci_degenerate_cases() -> None:
    assert paired_bootstrap_ci([], []) == (0.0, 0.0)
    assert paired_bootstrap_ci([1.0, 0.0, 2.0], [1.0, 0.0, 2.0], n_boot=500) == (0.0, 0.0)
    assert paired_bootstrap_ci([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], n_boot=500) == (3.0, 3.0)  # every resample totals +3
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


# ----------------------------------------------------------------------------- compare


def test_compare_math(eval_dirs: tuple[Path, Path]) -> None:
    base_dir, run_dir_ = eval_dirs
    cmp = write_compare(run_dir_, base_dir)
    assert json.loads((run_dir_ / "compare.json").read_text(encoding="utf-8")) == cmp
    assert (cmp["run_name"], cmp["base_run"], cmp["fingerprint_match"], cmp["eval_limit"]) == ("run1", "base", True, None)
    assert (cmp["primary_set"], cmp["primary_variant"]) == ("heldout", "canonical")

    h = cmp["headline"]
    assert (h["base_points"], h["run_points"], h["max_points"], h["delta_points"]) == (3.0, 7.0, 9.0, 4.0)
    assert h["delta_score_pct"] == pytest.approx(44.4444)
    assert (h["base_parse_fail_rate"], h["run_parse_fail_rate"]) == (0.1667, 0.0)
    assert h["delta_lenient_points"] == 3.0  # base lenient 4 (h6 lenient-only) -> run 7

    heldout_sets = cmp["sets"]["heldout"]
    assert heldout_sets["variants"]["bare"]["delta_points"] == -1.0
    assert heldout_sets["loglik"] == {"base_acc": 0.5, "run_acc": 0.6, "delta_acc_pct": 10.0}
    ext_canon = cmp["sets"][EXT]["variants"]["canonical"]
    assert (ext_canon["base_score_pct"], ext_canon["run_score_pct"], ext_canon["delta_score_pct"]) == (50.0, 75.0, 25.0)
    assert cmp["sets"][EXT]["loglik"]["delta_acc_pct"] == -2.0
    assert cmp["eval_sets"] == [{"name": "heldout", "variants": ["canonical", "bare", "payload_only"], "loglik": True},
                                {"name": EXT, "variants": ["canonical"], "loglik": True}]

    assert cmp["per_subject"] == {
        "biologia": {"base_points": 2.0, "run_points": 3.0, "max_points": 4.0, "delta_points": 1.0},
        "historia": {"base_points": 1.0, "run_points": 3.0, "max_points": 3.0, "delta_points": 2.0},
        "matematyka": {"base_points": 0.0, "run_points": 1.0, "max_points": 2.0, "delta_points": 1.0},
    }
    p = cmp["paired"]
    assert (p["wins"], p["losses"], p["ties"], p["n_paired"]) == (3, 1, 2, 6)
    assert p["sign_test_p"] == pytest.approx(0.625)
    lo, hi = p["bootstrap_ci95_delta_points"]
    assert lo <= h["delta_points"] <= hi
    assert cmp["flipped_items"] == {"gained": ["h2", "h4", "h5"], "lost": ["h3"]}
    assert cmp["general_heldout"] == {"base_nll": 1.25, "run_nll": 1.30, "rel_change": pytest.approx(0.04)}


def test_compare_refuses_fingerprint_mismatch_with_part_diff(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    base_dir, run_dir_ = eval_dirs
    parts = json.loads(json.dumps(FP_PARTS))
    parts["sets"][0]["sha256"] = "d" * 64
    parts["device"] = "mps"
    write_eval(run_dir_, "run1", RUN_RECORDS, loglik={"heldout": 0.6, EXT: 0.38}, nll=1.30, model="x", fingerprint="fp-2", parts=parts)

    with pytest.raises(ValueError, match="fingerprint mismatch") as err:
        compare_runs(base_dir, run_dir_)
    assert "sets.heldout.sha256" in str(err.value) and "device: base='cuda' run='mps'" in str(err.value)
    assert not (run_dir_ / "compare.json").exists()

    cmp = compare_runs(base_dir, run_dir_, allow_fingerprint_mismatch=True)
    assert cmp["fingerprint_match"] is False
    assert cmp["fingerprint_diff"] == {"device": {"base": "cuda", "run": "mps"},
                                       "sets.heldout.sha256": {"base": "a" * 64, "run": "d" * 64}}
    passed, reasons = registry.evaluate_gate(cmp, None, GATE)
    assert not passed and registry.failed_rules(reasons) == ["fp"]


def test_compare_without_predictions_and_general_nll(cfg: dict) -> None:
    base, run = run_dir(cfg, "base"), run_dir(cfg, "run1")
    write_eval(base, "base", BASE_RECORDS, loglik={}, nll=None, model="b", predictions=False)
    write_eval(run, "run1", RUN_RECORDS, loglik={}, nll=1.0, model="r", predictions=False)
    cmp = compare_runs(base, run)
    assert cmp["paired"] is None and cmp["flipped_items"] is None and cmp["general_heldout"] is None
    assert cmp["sets"]["heldout"]["loglik"] is None
    text = format_compare_table(cmp)
    assert "paired: n/a" in text and "general heldout NLL: n/a" in text


def test_compare_missing_summary_is_clear(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    with pytest.raises(FileNotFoundError, match="no summary.json"):
        compare_runs(eval_dirs[0], run_dir(cfg, "nope"))


def test_format_compare_table_order(eval_dirs: tuple[Path, Path]) -> None:
    text = format_compare_table(compare_runs(*eval_dirs))
    lines = text.splitlines()
    assert lines[0].startswith("run1 vs base") and lines[1].startswith("HEADLINE")
    markers = ["HEADLINE", "heldout/canonical *", "heldout/bare", "secondary sets", f"{EXT}/canonical", "loglik accuracy",
               "general heldout NLL", "per subject", "bootstrap CI95"]
    positions = [text.index(m) for m in markers]
    assert positions == sorted(positions)
    assert "gained: h2, h4, h5" in text and "lost:   h3" in text


def test_compare_organizer_block_and_table(eval_dirs: tuple[Path, Path]) -> None:
    """Organizer protocol (first-token letter argmax): deltas per set, only where BOTH sides measured it."""
    base_dir, run_dir_ = eval_dirs
    add_organizer(base_dir, {"heldout": organizer_block(0.50, abc_mass=0.80, skipped=2, rotated_accuracy=0.50),
                             EXT: organizer_block(0.40)})
    add_organizer(run_dir_, {"heldout": organizer_block(0.75, abc_mass=0.95, skipped=2, rotated_accuracy=0.625)})

    cmp = compare_runs(base_dir, run_dir_)
    assert cmp["sets"]["heldout"]["organizer"] == {
        "base_accuracy": 0.5, "run_accuracy": 0.75, "delta_acc_pct": 25.0, "base_mean_abc_mass": 0.80,
        "run_mean_abc_mass": 0.95, "delta_mean_abc_mass": 0.15, "rotated_delta_acc_pct": 12.5,
    }
    assert "organizer" not in cmp["sets"][EXT], "measured on one side only: the key is absent, not null"
    assert cmp["sets"]["heldout"]["variants"]["canonical"]["delta_points"] == 4.0, "existing keys are untouched"

    text = format_compare_table(cmp)
    assert "organizer (first-token letter argmax)" in text
    assert "50.00% ->  75.00%  +25.00 pp  ABC mass 0.800 -> 0.950  rotated +12.50 pp" in text
    assert text.index("loglik accuracy") < text.index("organizer (first") < text.index("general heldout NLL")

    for out in (base_dir, run_dir_):  # no organizer anywhere (e.g. an older summary): no section, no crash
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        for set_name in summary["sets"]:
            summary["sets"][set_name].pop("organizer", None)
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    cmp = compare_runs(base_dir, run_dir_)
    assert all("organizer" not in d for d in cmp["sets"].values())
    assert "organizer" not in format_compare_table(cmp)


# ----------------------------------------------------------------------------- gate


def make_cmp(run_name: str = "run1", delta: float = 3.0, *, base_pf: float = 0.0, run_pf: float = 0.0,
             nll_rel: float | None = 0.01, bare_delta: float = 0.0, ext_delta_pct: float = 0.0,
             loglik_delta_pct: float = 0.0, fp_match: bool = True, eval_limit: int | None = None, fingerprint: str = "fp-1",
             lenient_delta: float | None = None, ext_lenient_points: float = 0.0) -> dict:
    """compare.json-shaped dict with one knob per gate rule (defaults pass every rule). Sets, variants and loglik mirror
    configs/base.yaml eval.sets, so the comparison covers gate [scope]."""
    def variant(d: float, *, score_delta: float | None = None, bpf: float = 0.0, rpf: float = 0.0, lenient: float | None = None,
                max_points: float = 70.0) -> dict:
        return {
            "base_points": 30.0, "run_points": 30.0 + d, "max_points": max_points, "delta_points": d,
            "base_score_pct": round(100 * 30.0 / max_points, 2), "run_score_pct": round(100 * (30.0 + d) / max_points, 2),
            "delta_score_pct": round(100 * d / max_points, 4) if score_delta is None else score_delta,
            "base_parse_fail_rate": bpf, "run_parse_fail_rate": rpf, "delta_lenient_points": d if lenient is None else lenient,
        }

    primary = variant(delta, bpf=base_pf, rpf=run_pf, lenient=lenient_delta)
    headline_keys = ("base_points", "run_points", "max_points", "delta_points", "delta_score_pct",
                     "base_parse_fail_rate", "run_parse_fail_rate", "delta_lenient_points")
    return {
        "run_name": run_name, "base_run": "base", "fingerprint": fingerprint, "fingerprint_match": fp_match,
        "fingerprint_diff": {} if fp_match else {"device": {"base": "cuda", "run": "mps"}}, "eval_limit": eval_limit,
        "eval_sets": [{"name": "heldout", "variants": ["canonical", "bare", "payload_only"], "loglik": True},
                      {"name": EXT, "variants": ["canonical"], "loglik": True}],
        "primary_set": "heldout", "primary_variant": "canonical",
        "headline": {k: primary[k] for k in headline_keys},
        "sets": {
            "heldout": {"variants": {"canonical": primary, "bare": variant(bare_delta), "payload_only": variant(0.0)},
                        "loglik": {"base_acc": 0.5, "run_acc": 0.5 + loglik_delta_pct / 100, "delta_acc_pct": loglik_delta_pct}},
            EXT: {"variants": {"canonical": variant(0.0, score_delta=ext_delta_pct, lenient=ext_lenient_points, max_points=374.0)},
                  "loglik": {"base_acc": 0.4, "run_acc": 0.4, "delta_acc_pct": 0.0}},
        },
        "per_subject": {},
        "paired": {"wins": 3, "losses": 1, "ties": 56, "sign_test_p": 0.625, "bootstrap_ci95_delta_points": [-1.0, 5.0]},
        "general_heldout": None if nll_rel is None else {"base_nll": 1.0, "run_nll": 1.0 + nll_rel, "rel_change": nll_rel},
        "flipped_items": {"gained": [], "lost": []},
    }


BASE_CANDIDATE = {"run_name": "base", "delta_points": 0.0}


@pytest.mark.parametrize(("rule", "passing", "failing"), [
    ("1", {"delta": 0.5}, {"delta": 0.4}),
    ("2", {"base_pf": 0.05, "run_pf": 0.05}, {"base_pf": 0.05, "run_pf": 0.0667}),
    ("3", {"nll_rel": 0.05}, {"nll_rel": 0.0501}),
    ("4", {"bare_delta": -1.0}, {"bare_delta": -1.5}),
    ("5", {"ext_delta_pct": -2.0}, {"ext_delta_pct": -2.01}),
    ("6", {"loglik_delta_pct": -3.0}, {"loglik_delta_pct": -3.5}),
    ("fp", {"fp_match": True}, {"fp_match": False}),
    ("limit", {"eval_limit": None}, {"eval_limit": 6}),
    ("5b", {"delta": 0.7, "ext_delta_pct": -1.0}, {"delta": 0.6, "ext_delta_pct": -1.0}),  # 0.7 pts = +1.0 pp on 70
    ("8", {"lenient_delta": 0.0}, {"lenient_delta": -0.5}),
])
def test_gate_rule_pass_and_fail(rule: str, passing: dict, failing: dict) -> None:
    passed, reasons = registry.evaluate_gate(make_cmp(**passing), BASE_CANDIDATE, GATE)
    assert passed and registry.failed_rules(reasons) == []
    assert all(r.startswith(("PASS [", "FAIL [")) for r in reasons)
    assert any(r.startswith(f"PASS [{rule}]") for r in reasons)

    passed, reasons = registry.evaluate_gate(make_cmp(**failing), BASE_CANDIDATE, GATE)
    assert not passed and registry.failed_rules(reasons) == [rule]


def test_gate_missing_general_nll_fails() -> None:
    passed, reasons = registry.evaluate_gate(make_cmp(nll_rel=None), BASE_CANDIDATE, GATE)
    assert not passed and registry.failed_rules(reasons) == ["3"]
    assert any(r.startswith("FAIL [3]") and "missing" in r for r in reasons)


def test_gate_rule7_must_beat_current_candidate() -> None:
    cmp = make_cmp("run2", 3.0)
    assert registry.evaluate_gate(cmp, {"run_name": "run1", "delta_points": 2.5}, GATE)[0]
    passed, reasons = registry.evaluate_gate(cmp, {"run_name": "run1", "delta_points": 3.0}, GATE)  # a tie is not a win
    assert not passed and registry.failed_rules(reasons) == ["7"]
    assert registry.evaluate_gate(cmp, {"run_name": "run1", "delta_points": 5.0}, {**GATE, "must_beat_current_candidate": False})[0]
    assert registry.evaluate_gate(cmp, {"run_name": "run2", "delta_points": 3.0}, GATE)[0]  # re-recording the candidate itself

    margin = {**GATE, "min_improvement_points": 1.0}
    assert not registry.evaluate_gate(cmp, {"run_name": "run1", "delta_points": 2.5}, margin)[0]  # +0.5 is noise, not a win
    assert registry.evaluate_gate(make_cmp("run2", 3.5), {"run_name": "run1", "delta_points": 2.5}, margin)[0]


def test_gate_rule7_refuses_deltas_measured_under_different_fingerprints() -> None:
    """A candidate's delta measured under another eval (edited heldout, other grader code, other GPU) is on another scale."""
    stale = {"run_name": "run1", "delta_points": 3.0, "fingerprint": "fp-1", "model_path": "checkpoints/run1"}
    passed, reasons = registry.evaluate_gate(make_cmp("run2", 3.5, fingerprint="fp-2"), stale, GATE)
    assert not passed and registry.failed_rules(reasons) == ["7"]
    assert "not comparable" in next(r for r in reasons if r.startswith("FAIL [7]"))
    assert "python scripts/compare.py --model checkpoints/run1 --run-name run1" in next(r for r in reasons if r.startswith("FAIL [7]"))
    assert registry.evaluate_gate(make_cmp("run2", 3.5, fingerprint="fp-1"), stale, GATE)[0]
    base = {"run_name": "base", "delta_points": 0.0, "fingerprint": "fp-1"}  # the base's delta is 0 under any fingerprint
    assert registry.evaluate_gate(make_cmp("run2", 3.5, fingerprint="fp-2"), base, GATE)[0]
    assert registry.evaluate_gate(make_cmp("run1", 1.0, fingerprint="fp-2"), stale, GATE)[0], "re-recording the candidate refreshes it"


def test_gate_scope_refuses_narrowed_evals() -> None:
    """Removing eval sets, variants or loglik must not make rules 4-6 pass vacuously."""
    scope = {"heldout": {"variants": ["canonical", "bare", "payload_only"], "loglik": True}, EXT: {"variants": ["canonical"], "loglik": True}}
    full = make_cmp("run3", 4.0)
    passed, reasons = registry.evaluate_gate(full, BASE_CANDIDATE, GATE, required_scope=scope)
    assert passed and any(r.startswith("PASS [scope]") for r in reasons)

    narrowed = [
        [{"name": "heldout", "variants": ["canonical"], "loglik": False}],  # --set eval.sets=[{path: data/eval/heldout.jsonl}]
        [{"name": "heldout", "variants": ["canonical", "bare", "payload_only"], "loglik": True}],  # ext dropped
        [{"name": "heldout", "variants": ["canonical"], "loglik": True}, {"name": EXT, "variants": ["canonical"], "loglik": True}],
        [{"name": "heldout", "variants": ["canonical", "bare", "payload_only"], "loglik": False}, {"name": EXT, "variants": ["canonical"], "loglik": True}],
    ]
    for eval_sets in narrowed:
        passed, reasons = registry.evaluate_gate({**full, "eval_sets": eval_sets}, BASE_CANDIDATE, GATE, required_scope=scope)
        assert not passed and registry.failed_rules(reasons) == ["scope"], eval_sets
    assert "not compared: heldout/loglik" in next(r for r in reasons if r.startswith("FAIL [scope]"))

    legacy = {k: v for k, v in full.items() if k != "eval_sets"}  # older compare.json: coverage read from its sets
    legacy["sets"] = {"heldout": {**full["sets"]["heldout"], "loglik": None}}
    passed, reasons = registry.evaluate_gate(legacy, BASE_CANDIDATE, GATE, required_scope=scope)
    assert not passed and registry.failed_rules(reasons) == ["scope"]


def test_gate_lenient_and_net_score_rules() -> None:
    """A format gain must not hide lost knowledge (rule 8); secondary drops must be paid for by primary gains (5b)."""
    passed, reasons = registry.evaluate_gate(make_cmp(delta=8.0, lenient_delta=-5.0), BASE_CANDIDATE, GATE)
    assert not passed and registry.failed_rules(reasons) == ["8"]
    passed, reasons = registry.evaluate_gate(make_cmp(delta=8.0, lenient_delta=0.0), BASE_CANDIDATE, GATE)
    assert passed, "a pure format gain is not worse than base under any grader"
    passed, reasons = registry.evaluate_gate(make_cmp(delta=3.0, ext_lenient_points=-12.0), BASE_CANDIDATE, GATE)  # -3.2 pp on ext
    assert not passed and registry.failed_rules(reasons) == ["8"] and any(f"FAIL [8] {EXT}/canonical" in r for r in reasons)


def test_base_yaml_gate_rejects_noise_level_gain_with_real_exam_loss() -> None:
    """heldout +1 pt (3 wins / 2 losses) with -7 of 374 real CKE items passed every rule under the old thresholds."""
    gate = load_config(None)["gate"]
    cmp = make_cmp("run4", 1.0, ext_delta_pct=-1.87, ext_lenient_points=-7.0)
    passed, reasons = registry.evaluate_gate(cmp, BASE_CANDIDATE, gate)
    assert not passed and {"1", "5", "5b"} <= set(registry.failed_rules(reasons))
    passed, reasons = registry.evaluate_gate(make_cmp("run5", 2.0), {"run_name": "run4", "delta_points": 1.5, "fingerprint": "fp-1"}, gate)
    assert not passed and registry.failed_rules(reasons) == ["7"], "one more flip than the candidate is not an improvement"
    assert registry.evaluate_gate(make_cmp("run5", 2.5), {"run_name": "run4", "delta_points": 1.5, "fingerprint": "fp-1"}, gate)[0]


def with_organizer(cmp: dict, **deltas: float) -> dict:
    """compare.json sets.<set>.organizer blocks (eval.compare._organizer_delta output) on a copy of cmp."""
    out = json.loads(json.dumps(cmp))
    for set_name, delta in deltas.items():
        out["sets"][set_name]["organizer"] = {
            "base_accuracy": 0.5, "run_accuracy": round(0.5 + delta / 100, 4), "delta_acc_pct": delta,
            "base_mean_abc_mass": 0.80, "run_mean_abc_mass": 0.92, "delta_mean_abc_mass": 0.12,
            "rotated_delta_acc_pct": delta - 1.0,
        }
    return out


def test_gate_rule9_organizer_accuracy_pass_fail_and_skip() -> None:
    """Rule 9 guards the organizers' own protocol; a set nobody measured that way yields no line at all."""
    gate = {**GATE, "max_organizer_acc_drop_pct": 3.0}
    passed, reasons = registry.evaluate_gate(with_organizer(make_cmp(), heldout=-3.0), BASE_CANDIDATE, gate)
    assert passed and any(r.startswith("PASS [9] heldout organizer") for r in reasons)

    passed, reasons = registry.evaluate_gate(with_organizer(make_cmp(), heldout=-3.01), BASE_CANDIDATE, gate)
    assert not passed and registry.failed_rules(reasons) == ["9"]
    assert "-3.01 pp < -3.00" in next(r for r in reasons if r.startswith("FAIL [9]"))

    both = with_organizer(make_cmp(), heldout=1.0, **{EXT: -9.0})
    passed, reasons = registry.evaluate_gate(both, BASE_CANDIDATE, gate)
    assert not passed and registry.failed_rules(reasons) == ["9"] and any(f"FAIL [9] {EXT}" in r for r in reasons)

    passed, reasons = registry.evaluate_gate(make_cmp(), BASE_CANDIDATE, gate)  # nothing measured -> rule skipped
    assert passed and not any("[9]" in r for r in reasons)

    no_key = registry.evaluate_gate(with_organizer(make_cmp(), heldout=-3.5), BASE_CANDIDATE, GATE)  # older gate config
    assert not no_key[0] and registry.failed_rules(no_key[1]) == ["9"], "falls back to the 3.0 pp default"
    assert registry.evaluate_gate(with_organizer(make_cmp(), heldout=-10.0), BASE_CANDIDATE,
                                  {**GATE, "max_organizer_acc_drop_pct": 20.0})[0]


def test_base_yaml_defines_the_rule9_threshold() -> None:
    assert load_config(None)["gate"]["max_organizer_acc_drop_pct"] == 3.0


def test_reference_scope_follows_base_yaml_and_drops_missing_optional_sets() -> None:
    """[scope] asks for what configs/base.yaml evaluates; an optional set whose file is missing (e.g. the gitignored
    prawko splits on a fresh clone) is not required."""
    scope = registry.reference_eval_scope()
    assert scope["heldout"] == {"variants": ["canonical", "bare", "payload_only"], "loglik": True}
    assert not {p.stem for p in PRAWKO_EVAL_FILES if not p.exists()} & set(scope)
    passed, reasons = registry.evaluate_gate(make_cmp("run3", 4.0), BASE_CANDIDATE, GATE, required_scope=scope)
    assert passed and any(r.startswith("PASS [scope]") for r in reasons)


@needs_prawko_files
def test_reference_scope_ignores_organizer_only_eval_sets() -> None:
    """configs/base.yaml's prawko sets are scored only under the organizers' protocol (rule 9), so [scope] - which
    guards generated-answer variants and loglik - must not demand them."""
    scope = registry.reference_eval_scope()
    organizer_only = {name: spec for name, spec in scope.items() if not spec["variants"] and not spec["loglik"]}
    assert set(organizer_only) == {"ext_prawko_dev", "ext_prawko_test"}
    passed, reasons = registry.evaluate_gate(make_cmp("run3", 4.0), BASE_CANDIDATE, GATE, required_scope=scope)
    assert passed and any(r.startswith("PASS [scope]") for r in reasons)


def test_gate_requires_every_threshold_key() -> None:
    with pytest.raises(ValueError, match="min_delta_points"):
        registry.evaluate_gate(make_cmp(), BASE_CANDIDATE, {k: v for k, v in GATE.items() if k != "min_delta_points"})


# ----------------------------------------------------------------------------- registry


def record(cfg: dict, name: str, delta: float, *, tags: tuple[str, ...] = (), **cmp_kw) -> dict:
    ck = make_checkpoint(cfg, name)
    run_cfg = {**cfg, "run_name": name, "tags": list(tags)}
    return registry.record_run(
        run_cfg, run_dir=run_dir(run_cfg), cmp=make_cmp(name, delta, **cmp_kw),
        train_metrics={"n_train": 100, "epochs": [], "runtime_s": 1.0}, model_path=str(ck), adapter_path=str(ck / "adapter"),
    )


def candidate_name(cfg: dict) -> str:
    return registry.load_candidate(results_dir(cfg))["run_name"]


def test_ensure_candidate_base_fallback_and_score_refresh(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    rdir = results_dir(cfg)
    base_summary = rdir / "runs" / "base" / "summary.json"
    saved = base_summary.read_text(encoding="utf-8")
    base_summary.unlink()

    cand = registry.ensure_candidate(cfg)
    assert {k: cand[k] for k in ("run_name", "model_path", "delta_points", "score_pct", "tags", "reason")} == {
        "run_name": "base", "model_path": cfg["model"]["base"], "delta_points": 0.0, "score_pct": None,
        "tags": ["base-fallback"], "reason": "untouched base model (standing fallback)",
    }
    assert cand["promoted_at"] and json.loads((rdir / "CANDIDATE.json").read_text(encoding="utf-8")) == cand

    base_summary.write_text(saved, encoding="utf-8")
    assert registry.ensure_candidate(cfg)["score_pct"] == pytest.approx(33.33)
    assert registry.load_candidate(rdir)["score_pct"] == pytest.approx(33.33)

    record(cfg, "run1", 3.0)
    assert registry.ensure_candidate(cfg)["run_name"] == "run1"  # a promoted run is never reset to base


def test_worse_run_never_replaces_candidate_better_run_does(cfg: dict) -> None:
    assert record(cfg, "run1", 3.0)["promoted"] and candidate_name(cfg) == "run1"
    for name, delta, kw in [("worse", 2.0, {}), ("negative", -1.0, {}), ("bigger_but_broken", 9.0, {"nll_rel": 0.2}),
                            ("bigger_but_unparseable", 9.0, {"run_pf": 0.1}), ("no_general_nll", 9.0, {"nll_rel": None})]:
        row = record(cfg, name, delta, **kw)
        assert not row["gate_passed"] and not row["promoted"] and row["previous_candidate"] == "run1"
        assert candidate_name(cfg) == "run1"
    row = record(cfg, "better", 3.5)
    assert row["promoted"] and row["previous_candidate"] == "run1" and candidate_name(cfg) == "better"
    assert registry.load_candidate(results_dir(cfg))["delta_points"] == 3.5


def test_must_beat_current_candidate_ordering_across_three_runs(cfg: dict) -> None:
    a, b, c = record(cfg, "run_a", 2.0), record(cfg, "run_b", 1.5), record(cfg, "run_c", 2.5)
    assert (a["promoted"], b["promoted"], c["promoted"]) == (True, False, True)
    assert registry.failed_rules(b["gate_reasons"]) == ["7"]
    assert (a["previous_candidate"], b["previous_candidate"], c["previous_candidate"]) == ("base", "run_a", "run_a")
    assert candidate_name(cfg) == "run_c"


def test_tags_and_candidate_symlink_lifecycle(cfg: dict) -> None:
    rdir = results_dir(cfg)
    link = Path(cfg["paths"]["checkpoints_root"]) / f"CANDIDATE-{rdir.name}"
    assert registry.candidate_link(checkpoint_dir(cfg, "x").parent, rdir) == link
    record(cfg, "run1", 3.0, tags=("v1", "best"))
    assert registry.load_tags(rdir) == {"v1": "run1", "best": "run1"}
    assert link.is_symlink() and os.readlink(link) == "run1"
    assert link.resolve() == checkpoint_dir(cfg, "run1").resolve()

    record(cfg, "run2", 1.0, tags=("best",))  # fails rule 7: tags and link unchanged
    assert registry.load_tags(rdir) == {"v1": "run1", "best": "run1"} and os.readlink(link) == "run1"

    record(cfg, "run3", 4.0, tags=("best",))
    assert registry.load_tags(rdir) == {"v1": "run1", "best": "run3"} and os.readlink(link) == "run3"

    cand = registry.promote(cfg, "base", reason="roll back to the untouched base model")
    assert cand["run_name"] == "base" and cand["model_path"] == cfg["model"]["base"]
    assert not link.is_symlink() and not link.exists()
    assert registry.load_tags(rdir)["base-fallback"] == "base"


def test_leaderboard_has_base_row_and_candidate_star(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    rdir = results_dir(cfg)
    registry.ensure_candidate(cfg)
    assert "★ base" in registry.render_leaderboard(rdir)

    record(cfg, "run1", 3.0, tags=("v1",))
    record(cfg, "run2", -1.0)
    text = (rdir / "LEADERBOARD.md").read_text(encoding="utf-8")
    assert text == registry.render_leaderboard(rdir)
    header = text.split("| run |")[0]
    assert header.endswith("\n\n")  # a list item directly above would swallow the table in Markdown
    for column in ("Δpts", "score%", "parse-fail%", "Δlenient", "secondary Δ%", "loglik Δ%", "general NLL Δ%", "CI95", "gate"):
        assert column in header  # every column is explained above the table
    table = [line for line in text.splitlines() if line.startswith("| ")]
    assert table[0].split(" | ")[:3] == ["| run", "Δpts", "score%"]
    names = [line.split(" | ")[0][2:] for line in table[1:]]
    assert names == ["★ run1", "base", "run2"]  # sorted by headline delta: +3, 0, -1
    base_row = table[2]
    assert "| +0.00 | 33.33 |" in base_row
    assert "FAIL 1" in table[3] and "| pass |" in table[1] and "v1" in table[1]


def test_organizer_row_fields_and_leaderboard_column(cfg: dict) -> None:
    ck = make_checkpoint(cfg, "run1")
    cmp = with_organizer(make_cmp("run1", 3.0), heldout=2.0, **{EXT: -1.0})
    row = registry.record_run({**cfg, "run_name": "run1"}, run_dir=run_dir(cfg, "run1"), cmp=cmp, train_metrics=None,
                              model_path=str(ck), adapter_path=str(ck / "adapter"))
    assert row["promoted"] and "9" not in row["gate_failed_rules"]
    assert row["organizer_delta_acc_pct"] == {"heldout": 2.0, EXT: -1.0}
    assert row["organizer_abc_mass"] == {"heldout": 0.92, EXT: 0.92}

    record(cfg, "run2", 4.0)  # no organizer data: the keys stay, empty
    assert registry.load_runs(results_dir(cfg))[-1]["organizer_delta_acc_pct"] == {}

    text = (results_dir(cfg) / "LEADERBOARD.md").read_text(encoding="utf-8")
    header, table = text.split("| run |")[0], [line for line in text.splitlines() if line.startswith("| ")]
    assert "org Δ%" in header and table[0].split(" | ")[7] == "org Δ%"
    by_run = {line.split(" | ")[0][2:]: line.split(" | ") for line in table[1:]}
    assert by_run["★ run2"][7] == "—" and by_run["run1"][7] == f"heldout +2.00; {EXT} -1.00"
    assert by_run["base"][7] == "—"


def test_cleanup_deletes_merged_weights_but_keeps_adapter_and_candidate(cfg: dict) -> None:
    cfg = {**cfg, "pipeline": {**cfg["pipeline"], "keep_merged": "all"}}
    record(cfg, "run1", 3.0)
    record(cfg, "run2", 1.0)
    bare = checkpoint_dir(cfg, "merged_only")  # a recorded run without adapter/: its merged model is the only copy
    bare.mkdir(parents=True)
    (bare / "model.safetensors").write_text("{}", encoding="utf-8")
    registry.record_run({**cfg, "run_name": "merged_only"}, run_dir=run_dir(cfg, "merged_only"), cmp=make_cmp("merged_only", 0.0),
                        train_metrics=None, model_path=str(bare), adapter_path=None)
    rdir, ck_root = results_dir(cfg), Path(cfg["paths"]["checkpoints_root"])

    assert registry.cleanup_checkpoints(rdir, ck_root, "all") == []
    deleted = registry.cleanup_checkpoints(rdir, ck_root, "candidate_only")
    run2 = checkpoint_dir(cfg, "run2")
    assert sorted(deleted) == sorted(str(run2 / f) for f in MERGED_FILES)
    assert all((run2 / f).exists() for f in KEPT_FILES)
    assert all((checkpoint_dir(cfg, "run1") / f).exists() for f in MERGED_FILES + KEPT_FILES)
    assert (bare / "model.safetensors").exists()
    assert registry.cleanup_checkpoints(rdir, ck_root, "candidate_only") == []  # idempotent
    with pytest.raises(ValueError, match="keep_merged"):
        registry.cleanup_checkpoints(rdir, ck_root, "none")


def test_record_run_applies_keep_merged_policy_after_promotion(cfg: dict) -> None:
    record(cfg, "run1", 2.0)
    record(cfg, "run2", 3.0)  # promoted: the previous candidate run1 loses its merged weights, keeps its adapter
    run1, run2 = checkpoint_dir(cfg, "run1"), checkpoint_dir(cfg, "run2")
    assert not any((run1 / f).exists() for f in MERGED_FILES) and all((run1 / f).exists() for f in KEPT_FILES)
    assert all((run2 / f).exists() for f in MERGED_FILES + KEPT_FILES)


def test_promote_failed_run_requires_force_and_is_logged(cfg: dict) -> None:
    record(cfg, "run1", 3.0)
    record(cfg, "run2", 1.0)
    rdir = results_dir(cfg)
    with pytest.raises(ValueError, match="failed the promotion gate"):
        registry.promote(cfg, "run2", reason="looks better on bare")
    with pytest.raises(ValueError, match="reason"):
        registry.promote(cfg, "run2", reason=" ", force=True)
    with pytest.raises(ValueError, match="no recorded row"):
        registry.promote(cfg, "ghost", reason="x", force=True)
    assert candidate_name(cfg) == "run1"

    cand = registry.promote(cfg, "run2", reason="looks better on bare", tags=["manual"], force=True)
    assert (cand["run_name"], cand["forced"], cand["manual"], cand["tags"]) == ("run2", True, True, ["manual"])
    assert cand["warnings"]  # run2's merged weights were cleaned when it failed the gate
    assert f"python scripts/merge_and_export.py --adapter {checkpoint_dir(cfg, 'run2').resolve() / 'adapter'}" in cand["warnings"][0]
    assert candidate_name(cfg) == "run2" and registry.load_tags(rdir)["manual"] == "run2"
    event = registry.load_runs(rdir)[-1]
    assert {k: event[k] for k in ("event", "run_name", "force", "forced", "gate_passed", "previous_candidate", "reason")} == {
        "event": "manual_promote", "run_name": "run2", "force": True, "forced": True, "gate_passed": False,
        "previous_candidate": "run1", "reason": "looks better on bare",
    }
    assert registry.promote(cfg, "run1", reason="back to the gated run")["forced"] is False  # passed its gate: no force


def test_runs_jsonl_is_append_only(cfg: dict) -> None:
    path = results_dir(cfg) / "runs.jsonl"
    record(cfg, "run1", 3.0)
    first = path.read_bytes()
    record(cfg, "run2", 1.0)
    second = path.read_bytes()
    registry.promote(cfg, "run1", reason="pin", tags=["final"])
    third = path.read_bytes()
    record(cfg, "run1", 3.0)  # re-recording appends, never rewrites
    fourth = path.read_bytes()
    assert second.startswith(first) and third.startswith(second) and fourth.startswith(third)
    rows = [json.loads(line) for line in fourth.decode().splitlines()]
    assert [(r["event"], r["run_name"]) for r in rows] == [("run", "run1"), ("run", "run2"), ("manual_promote", "run1"), ("run", "run1")]
    assert rows[-1]["gate_passed"] and candidate_name(cfg) == "run1"


def test_record_run_without_train_metrics_reads_checkpoint_artifacts(cfg: dict, tmp_path: Path) -> None:
    run_cfg = load_config(None, {
        "run_name": "run7", "tags": ["exp"], "train.learning_rate": 1e-4, "lora.r": 32, "lora.alpha": 64, "train.epochs": 3,
        "data.general_ratio": 0.3, "data.synthetic": str(tmp_path / "synthetic" / "clean.jsonl"),
        "paths.results_root": cfg["paths"]["results_root"], "paths.checkpoints_root": cfg["paths"]["checkpoints_root"],
    })
    ck = make_checkpoint(cfg, "run7")
    dump_config(run_cfg, ck / "resolved_config.yaml")
    (ck / "train_metrics.json").write_text(json.dumps({"n_train": 123, "runtime_s": 9.5, "epochs": [{"epoch": 1, "train_loss": 0.5}]}))
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "synthetic" / "dedup_report.json").write_text(json.dumps({
        "dropped": {"heldout": 3, "fuzzy": 1}, "thresholds": {"thr_qa": 0.8}, "examples": {"heldout": [{"candidate": "x"}]},
    }))

    compare_only_cfg = {**cfg, "run_name": "base"}  # e.g. scripts/compare.py --run run7 without --config
    row = registry.record_run(compare_only_cfg, run_dir=run_dir(cfg, "run7"), cmp=make_cmp("run7", 2.0), train_metrics=None,
                              model_path=str(ck), adapter_path=None)
    assert row["hparams"] == {"lr": 1e-4, "r": 32, "alpha": 64, "epochs": 3, "general_ratio": 0.3, "general_ratio_trained": None,
                              "variant_mix": run_cfg["data"]["variant_mix"], "n_train": 123}
    assert row["config_hash"] == config_hash(run_cfg) and row["tags"] == ["exp"]
    assert row["train"]["runtime_s"] == 9.5 and row["train"]["last_epoch"] == {"epoch": 1, "train_loss": 0.5}
    assert row["dedup"] == {"dropped": {"heldout": 3, "fuzzy": 1}, "thresholds": {"thr_qa": 0.8}}
    assert (row["delta_points"], row["ci95"], row["wins"], row["general_nll_rel_change"]) == (2.0, [-1.0, 5.0], 3, 0.01)


def test_candidate_failing_re_evaluation_is_demoted_to_base(cfg: dict) -> None:
    """E.g. heldout fixed / grader changed: run1 re-measured at -2 must not stay the submission (nor stay rule 7's bar)."""
    rdir = results_dir(cfg)
    link = registry.candidate_link(Path(cfg["paths"]["checkpoints_root"]), rdir)
    assert record(cfg, "run1", 3.0)["promoted"] and os.readlink(link) == "run1"

    row = record(cfg, "run1", -2.0, bare_delta=-2.0)
    assert row["demoted"] and not row["promoted"] and {"1", "4"} <= set(row["gate_failed_rules"])
    cand = registry.load_candidate(rdir)
    assert (cand["run_name"], cand["model_path"], cand["delta_points"]) == ("base", cfg["model"]["base"], 0.0)
    assert "failed re-evaluation" in cand["reason"] and not link.is_symlink()
    assert "★ base" in (rdir / "LEADERBOARD.md").read_text(encoding="utf-8") and "(demoted)" in registry.render_leaderboard(rdir)

    assert record(cfg, "run2", 2.5)["promoted"] and candidate_name(cfg) == "run2", "the stale +3 no longer blocks a better run"


def test_re_record_failing_only_non_verdict_rules_keeps_the_candidate(cfg: dict) -> None:
    record(cfg, "run1", 3.0)
    for kw in ({"fp_match": False}, {"eval_limit": 6}, {"eval_limit": 6, "bare_delta": -5.0}):
        row = record(cfg, "run1", -2.0, **kw)
        assert not row["promoted"] and not row["demoted"] and candidate_name(cfg) == "run1", kw
    assert registry.load_candidate(results_dir(cfg))["delta_points"] == 3.0


def test_record_run_refuses_another_model_under_the_candidates_run_name(cfg: dict) -> None:
    rdir = results_dir(cfg)
    link = registry.candidate_link(Path(cfg["paths"]["checkpoints_root"]), rdir)
    record(cfg, "run1", 5.0)
    before = (rdir / "CANDIDATE.json").read_text(encoding="utf-8")
    other = make_checkpoint(cfg, "run2")
    with pytest.raises(ValueError, match="belongs to the current candidate"):
        registry.record_run({**cfg, "run_name": "run1"}, run_dir=run_dir(cfg, "run1"), cmp=make_cmp("run1", 1.0),
                            train_metrics=None, model_path=str(other), adapter_path=str(other / "adapter"))
    assert (rdir / "CANDIDATE.json").read_text(encoding="utf-8") == before and os.readlink(link) == "run1"
    assert len(registry.load_runs(rdir)) == 1
    assert all((checkpoint_dir(cfg, "run1") / f).exists() for f in MERGED_FILES), "the real candidate's weights are kept"
    with pytest.raises(ValueError, match="belongs to the current candidate"):
        registry.check_run_name_owner(cfg, "run1", other)
    registry.check_run_name_owner(cfg, "run1", checkpoint_dir(cfg, "run1") / "adapter")  # same checkpoint: allowed

    row = registry.record_run({**cfg, "run_name": "run1"}, run_dir=run_dir(cfg, "run1"), cmp=make_cmp("run1", 5.5), train_metrics=None,
                              model_path=str(checkpoint_dir(cfg, "run1") / "adapter"), adapter_path=None)
    assert row["promoted"] and candidate_name(cfg) == "run1", "re-recording the candidate through its adapter/ is legitimate"


def test_record_run_scope_never_promotes_a_narrowed_eval(cfg: dict) -> None:
    make_checkpoint(cfg, "run3")
    narrowed = {**make_cmp("run3", 4.0), "eval_sets": [{"name": "heldout", "variants": ["canonical"], "loglik": False}]}
    row = registry.record_run({**cfg, "run_name": "run3"}, run_dir=run_dir(cfg, "run3"), cmp=narrowed, train_metrics=None,
                              model_path=str(checkpoint_dir(cfg, "run3")), adapter_path=None)
    assert not row["promoted"] and row["gate_failed_rules"] == ["scope"] and candidate_name(cfg) == "base"


def test_record_run_warns_when_the_promoted_checkpoint_has_no_merged_model(cfg: dict) -> None:
    ck = checkpoint_dir(cfg, "run1")
    (ck / "adapter").mkdir(parents=True)
    (ck / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    row = registry.record_run({**cfg, "run_name": "run1"}, run_dir=run_dir(cfg, "run1"), cmp=make_cmp("run1", 3.0), train_metrics=None,
                              model_path=str(ck), adapter_path=str(ck / "adapter"))
    assert row["promoted"]
    warnings = registry.load_candidate(results_dir(cfg))["warnings"]
    assert len(warnings) == 1 and "python scripts/merge_and_export.py --adapter" in warnings[0]
    make_checkpoint(cfg, "run2")
    record(cfg, "run2", 4.0)
    assert registry.load_candidate(results_dir(cfg))["warnings"] == []


def test_two_base_models_keep_separate_candidate_links(tmp_path: Path) -> None:
    """checkpoints/ is shared by every base model: one base's promotion or rollback must not touch the other's link."""
    roots = {"paths.results_root": str(tmp_path / "results"), "paths.checkpoints_root": str(tmp_path / "checkpoints"), "tags": []}
    small = load_config(None, {**roots, "run_name": "s1"})
    big = load_config(None, {**roots, "run_name": "x11b", "model.base": "speakleash/Bielik-11B-v2.6-Instruct"})
    for c in (small, big):
        c["gate"] = dict(GATE)
    ck_root = tmp_path / "checkpoints"
    record(small, "s1", 3.0)
    record(big, "x11b", 2.0)
    small_link, big_link = registry.candidate_link(ck_root, results_dir(small)), registry.candidate_link(ck_root, results_dir(big))
    assert small_link != big_link and os.readlink(small_link) == "s1" and os.readlink(big_link) == "x11b"
    assert registry.protected_checkpoints(results_dir(big), ck_root) == {"s1", "x11b"}

    registry.promote(big, "base", reason="roll back 11B")
    assert not big_link.is_symlink() and os.readlink(small_link) == "s1"
    assert registry.protected_checkpoints(results_dir(big), ck_root) == {"s1"}


def test_compare_refuses_a_base_dir_without_a_base_eval(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    """run_eval.py --model checkpoints/run1 --run-name base used to put a fine-tuned eval into runs/base."""
    base_dir, run_dir_ = eval_dirs
    summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    (base_dir / "summary.json").write_text(json.dumps({**summary, "is_base": False, "model": "checkpoints/run1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not hold a base-model eval"):
        compare_runs(base_dir, run_dir_)


def test_compare_and_promote_point_to_the_right_base_model_without_config(cfg: dict) -> None:
    other_cfg = load_config(None, {"model.base": "speakleash/Bielik-11B-v2.6-Instruct", "run_name": "bielik11b-v0", "tags": [],
                                   "paths.results_root": cfg["paths"]["results_root"], "paths.checkpoints_root": cfg["paths"]["checkpoints_root"]})
    other_cfg["gate"] = dict(GATE)
    write_eval(run_dir(other_cfg, "base"), "base", BASE_RECORDS, loglik={"heldout": 0.5, EXT: 0.4}, nll=1.0, model="b")
    write_eval(run_dir(other_cfg, "bielik11b-v0"), "bielik11b-v0", RUN_RECORDS, loglik={"heldout": 0.5, EXT: 0.4}, nll=1.0, model="m")
    record(other_cfg, "bielik11b-v0", 3.0)
    sets = cli_overrides(cfg)
    res = run_script("compare.py", "--run", "bielik11b-v0", *sets)
    assert res.returncode == 1 and "bielik-11b-v2.6-instruct" in res.stderr and "--config" in res.stderr
    res = run_script("promote.py", "--run", "bielik11b-v0", "--reason", "x", *sets)
    assert res.returncode == 1 and "bielik-11b-v2.6-instruct" in res.stderr and "--config" in res.stderr
    assert not (results_dir(cfg) / "CANDIDATE.json").exists(), "nothing written into the default base model's registry"


def test_recorded_paths_are_repo_relative_inside_the_repo(tmp_path: Path) -> None:
    """results/ is shared between machines: rows and CANDIDATE.json must not carry one checkout's absolute path."""
    assert registry._portable(ROOT / "checkpoints" / "run1") == "checkpoints/run1"
    assert registry._portable("checkpoints/run1/adapter") == "checkpoints/run1/adapter"
    assert registry._local_path("checkpoints/run1") == ROOT / "checkpoints" / "run1"  # read back against the repo, not the cwd
    outside = tmp_path / "ck" / "run1"
    assert registry._portable(outside) == str(outside.resolve())


def test_record_run_refuses_the_base_run(cfg: dict) -> None:
    with pytest.raises(ValueError, match="base run"):
        registry.record_run(cfg, run_dir=run_dir(cfg, "base"), cmp={**make_cmp("base"), "base_run": "base"},
                            train_metrics=None, model_path=cfg["model"]["base"], adapter_path=None)


# ----------------------------------------------------------------------------- scripts


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], capture_output=True, text=True, cwd=ROOT, timeout=120)


def cli_overrides(cfg: dict) -> list[str]:
    out = ["--set", f"paths.results_root={cfg['paths']['results_root']}", "--set", f"paths.checkpoints_root={cfg['paths']['checkpoints_root']}",
           "--set", "pipeline.keep_merged=candidate_only"]
    for key, value in GATE.items():
        out += ["--set", f"gate.{key}={json.dumps(value)}"]
    return out


@pytest.mark.parametrize("script", ["compare.py", "status.py", "promote.py"])
def test_script_help(script: str) -> None:
    result = run_script(script, "--help")
    assert result.returncode == 0 and "usage:" in result.stdout


def test_compare_status_promote_scripts_end_to_end(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    sets = cli_overrides(cfg)
    ck = make_checkpoint(cfg, "run1")
    rdir = results_dir(cfg)
    link = registry.candidate_link(Path(cfg["paths"]["checkpoints_root"]), rdir)

    dry = run_script("compare.py", "--run", "run1", "--no-record", *sets)
    assert dry.returncode == 0, dry.stderr
    assert "HEADLINE" in dry.stdout and "dry run" in dry.stdout and not (rdir / "runs.jsonl").exists()

    res = run_script("compare.py", "--run", "run1", *sets)
    assert res.returncode == 0, res.stderr
    assert "GATE PASSED" in res.stdout and "CANDIDATE: run1" in res.stdout
    assert (eval_dirs[1] / "compare.json").exists() and os.readlink(link) == "run1"
    row = registry.load_runs(rdir)[-1]
    assert row["adapter_path"] == str(ck / "adapter") and row["hparams"]["n_train"] is None

    status = run_script("status.py", "--json", *sets)
    assert status.returncode == 0, status.stderr
    st = json.loads(status.stdout)[rdir.name]
    assert st["candidate"]["run_name"] == "run1" and [e["run_name"] for e in st["leaderboard"]] == ["run1", "base"]
    text = run_script("status.py", *sets)
    assert text.returncode == 0 and "CANDIDATE: run1" in text.stdout and "★ run1" in text.stdout

    back = run_script("promote.py", "--run", "base", "--reason", "roll back", "--tag", "safe", *sets)
    assert back.returncode == 0, back.stderr
    assert "CANDIDATE -> base" in back.stdout and not link.is_symlink()
    assert registry.load_tags(rdir)["safe"] == "base"


def test_compare_script_errors_exit_nonzero(cfg: dict, eval_dirs: tuple[Path, Path]) -> None:
    sets = cli_overrides(cfg)
    summary_path = eval_dirs[1] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(fingerprint="fp-2", fingerprint_parts={**FP_PARTS, "device": "mps"})
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    mismatch = run_script("compare.py", "--run", "run1", *sets)
    assert mismatch.returncode == 1 and "fingerprint mismatch" in mismatch.stderr and "device" in mismatch.stderr
    assert run_script("compare.py", "--run", "missing", *sets).returncode == 1
    assert run_script("compare.py", "--run", "run1", "--limit", "3").returncode == 2
    assert run_script("compare.py", "--model", "x", "--base-run", "other").returncode == 2
    assert run_script("promote.py", "--run", "run1", "--reason", "x", *sets).returncode == 1  # never recorded


def load_compare_script():
    spec = importlib.util.spec_from_file_location("compare_script", ROOT / "scripts" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_script_mode_b_evaluates_then_records(cfg: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_compare_script()
    ck = make_checkpoint(cfg, "run9")
    dump_config({**cfg, "model": {**cfg["model"], "base": "speakleash/Bielik-11B-v2.3-Instruct"}}, ck / "resolved_config.yaml")
    calls: list[tuple[str, str, str, int | None]] = []

    def fake_evaluate(run_cfg: dict, model: str, run_name: str, base_run: str, limit: int | None) -> None:
        calls.append((run_cfg["model"]["base"], run_name, base_run, limit))
        parts = {**FP_PARTS, "limit": limit if limit is not None else run_cfg["eval"]["limit"]}
        loglik = {"heldout": 0.5, EXT: 0.4}
        write_eval(run_dir(run_cfg, base_run), base_run, BASE_RECORDS, loglik=loglik, nll=1.0, model="b", parts=parts)
        write_eval(run_dir(run_cfg, run_name), run_name, RUN_RECORDS, loglik=loglik, nll=1.0, model=model, parts=parts)

    monkeypatch.setattr(script, "_evaluate", fake_evaluate)
    slug_dir = Path(cfg["paths"]["results_root"]) / "bielik-11b-v2.3-instruct"

    assert script.main(["--model", str(ck), "--limit", "3", *cli_overrides(cfg)]) == 0
    assert calls[-1] == ("speakleash/Bielik-11B-v2.3-Instruct", "run9-limit3", "base-limit3", 3)
    assert (slug_dir / "runs" / "run9-limit3" / "compare.json").exists() and not (slug_dir / "runs.jsonl").exists()

    # a CLI override that changes the eval fingerprint gets its own -dbg<fp> names: the full evals are never overwritten
    assert script.main(["--model", str(ck), "--set", "eval.limit=3", *cli_overrides(cfg)]) == 0
    _, run_name, base_run, limit = calls[-1]
    assert re.fullmatch(r"run9-dbg[0-9a-f]{8}", run_name) and base_run == "base" + run_name[len("run9"):] and limit == 3
    assert script.main(["--model", str(ck), "--set", "eval.max_new_tokens=8", *cli_overrides(cfg)]) == 0
    assert re.fullmatch(r"run9-dbg[0-9a-f]{8}", calls[-1][1]) and calls[-1][3] is None
    assert not (slug_dir / "runs.jsonl").exists(), "limited and -dbg evals are never recorded"

    limited_cfg = tmp_path / "limited.yaml"
    limited_cfg.write_text("eval:\n  limit: 3\n", encoding="utf-8")
    assert script.main(["--model", str(ck), "--config", str(limited_cfg), "--base-model", "speakleash/Bielik-11B-v2.3-Instruct",
                        *cli_overrides(cfg)]) == 0
    assert calls[-1][1:3] == ("run9", "base"), "a config-file eval.limit keeps plain names (and is still never recorded)"
    assert not (slug_dir / "runs.jsonl").exists()

    assert script.main(["--model", str(ck / "adapter"), *cli_overrides(cfg)]) == 0
    assert calls[-1] == ("speakleash/Bielik-11B-v2.3-Instruct", "run9", "base", None)
    assert json.loads((slug_dir / "CANDIDATE.json").read_text(encoding="utf-8"))["run_name"] == "run9"
    assert [r["run_name"] for r in registry.load_runs(slug_dir)] == ["run9"]

    other = make_checkpoint(cfg, "run10")
    with pytest.raises(ValueError, match="belongs to the current candidate"):  # refused BEFORE evaluating over runs/run9
        script.main(["--model", str(other), "--run-name", "run9", "--base-model", "speakleash/Bielik-11B-v2.3-Instruct", *cli_overrides(cfg)])
    assert calls[-1][1] == "run9" and len(calls) == 5


PRAWKO_ROWS = [  # two rows shaped exactly like the organizers' datasets/prawko-v2/data.json (note the leading spaces)
    {"id": "4367", "question": "Który z czynników ogranicza pole widzenia?",
     "options": [" Klimatyzacja.", "Lusterko wewnętrzne.", "Oślepiające światła."], "answer": 2, "points": 2,
     "english_question": "Which factor restricts the field of view?", "split": "dev"},
    {"id": "10840", "question": "Jak przewozisz dziecko niższe niż 150 cm?",
     "options": ["Na kolanach pasażera.", " W foteliku bezpieczeństwa.", "Tyłem do kierunku jazdy."], "answer": 1,
     "points": 3, "split": "dev"},
]


def load_import_prawko():
    spec = importlib.util.spec_from_file_location("import_prawko", ROOT / "scripts" / "import_prawko.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_prawko_converts_rows_to_items(tmp_path: Path) -> None:
    """Their int answer index -> our letter; option text stays verbatim (their grader renders it unchanged)."""
    from eval.schema import load_items, validate_item

    mod = load_import_prawko()
    items = mod.convert_split(PRAWKO_ROWS, "dev")
    assert [it["id"] for it in items] == ["prawko-4367", "prawko-10840"]
    assert [it["answer"] for it in items] == ["C", "B"]
    assert items[0]["options"] == {"A": " Klimatyzacja.", "B": "Lusterko wewnętrzne.", "C": "Oślepiające światła."}
    assert all(it["type"] == "mc" and it["subject"] == "wos" and it["points"] == 1 for it in items)
    assert [it["source"] for it in items] == ["prawko-v2:dev", "prawko-v2:dev"]
    assert all(validate_item(dict(it)) for it in items)
    assert mod.answer_counts(items) == {"A": 0, "B": 1, "C": 1}
    assert mod.source_points(PRAWKO_ROWS) == {"2": 1, "3": 1}  # reported only: their harness scores accuracy over rows

    with pytest.raises(ValueError, match="options"):
        mod.convert_row({"id": "1", "question": "q", "options": ["a", "b"], "answer": 0}, "dev")
    with pytest.raises(ValueError, match="answer"):
        mod.convert_row({"id": "1", "question": "q", "options": ["a", "b", "c"], "answer": 3}, "dev")

    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"train": PRAWKO_ROWS, "dev": PRAWKO_ROWS, "test": PRAWKO_ROWS}), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert mod.main(["--from-file", str(data_file), "--out-dir", str(out_dir), "--splits", "dev"]) == 0
    assert load_items(out_dir / "ext_prawko_dev.jsonl") == items
    assert not (out_dir / "ext_prawko_train.jsonl").exists(), "their train split is never imported"


@needs_prawko_files
def test_imported_prawko_eval_sets_match_the_organizers_splits() -> None:
    """The local (gitignored) output of scripts/import_prawko.py matches their manifest."""
    from eval.schema import load_items

    mod = load_import_prawko()
    for split, n, letters in (("dev", 25, {"A": 7, "B": 7, "C": 11}), ("test", 40, {"A": 12, "B": 13, "C": 15})):
        items = load_items(ROOT / "data" / "eval" / f"ext_prawko_{split}.jsonl")  # validates every item
        assert len(items) == n and mod.answer_counts(items) == letters  # counts of their manifest.json
        assert all(it["source"] == f"prawko-v2:{split}" and it["id"].startswith("prawko-") for it in items)


def test_checkpoint_eval_overrides_reproduce_the_pipeline_eval_settings(tmp_path: Path) -> None:
    from eval.runner import checkpoint_eval_overrides

    trained = load_config(ROOT / "configs" / "bielik11b.yaml")  # eval.batch_size 8, base.yaml says 16
    ck = tmp_path / "ck" / "bielik11b-v0"
    (ck / "adapter").mkdir(parents=True)
    dump_config(trained, ck / "resolved_config.yaml")
    overrides = checkpoint_eval_overrides(ck)
    assert checkpoint_eval_overrides(ck / "adapter") == overrides and checkpoint_eval_overrides(tmp_path) == []
    resolved = load_config(None, overrides)
    assert resolved["eval"] == trained["eval"] and resolved["eval"]["batch_size"] == 8
    assert load_config(None, overrides + ["eval.batch_size=4"])["eval"]["batch_size"] == 4, "CLI --set still wins"
