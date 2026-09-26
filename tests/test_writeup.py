import importlib.util
import json
import pathlib

from train.config import load_config, results_dir

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("writeup", ROOT / "scripts" / "writeup.py")
writeup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(writeup)


def _cfg(tmp_path):
    return load_config("configs/base.yaml", [f"paths.results_root={tmp_path}"])


def test_empty_registry_points_to_the_next_commands(tmp_path):
    text = writeup.build(_cfg(tmp_path))
    assert "## 1. Evaluation harness" in text and "60 original matura-style items" in text
    assert "run_eval.py --model speakleash/Bielik-4.5B-v3.0-Instruct --base" in text
    assert "configs/run1.yaml" in text


def test_runs_table_splits_format_and_knowledge_and_reports_gate(tmp_path):
    cfg = _cfg(tmp_path)
    rdir = results_dir(cfg)
    rdir.mkdir(parents=True)
    rows = [
        {"event": "run", "run_name": "run1", "timestamp": "2026-09-17T10:00:00+00:00", "notes": "v0-safe", "delta_points": 6.0,
         "delta_lenient_points": 2.0, "ci95": [2.0, 10.0], "parse_fail_rate": 0.0, "secondary_delta_score_pct": {"ext": 0.5},
         "general_nll_rel_change": 0.01, "gate_passed": True, "gate_failed_rules": [], "promoted": True,
         "hparams": {"lr": 1e-4, "r": 16, "epochs": 2, "general_ratio": 0.3}},
        {"event": "run", "run_name": "s1-nogeneral", "timestamp": "2026-09-17T12:00:00+00:00", "delta_points": 7.0,
         "delta_lenient_points": 1.0, "gate_passed": False, "gate_failed_rules": ["3"],
         "gate_reasons": ["PASS [1] ok", "FAIL [3] general NLL relative change +9.00% > max +5.00%"], "hparams": {}},
        {"event": "manual_promote", "run_name": "run1", "timestamp": "2026-09-17T13:00:00+00:00"},
    ]
    (rdir / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (rdir / "CANDIDATE.json").write_text(json.dumps({"run_name": "run1", "model_path": "checkpoints/run1", "delta_points": 6.0,
                                                     "score_pct": 60.0, "reason": "gate passed", "tags": ["v0-safe"]}))
    text = writeup.build(cfg)
    assert "| run1 | v0-safe |" in text and "+6.0 [+2.0, +10.0] | +4.0 | +2.0 |" in text and "ext +0.5" in text
    assert "**What worked**: `run1`" in text and "mostly answer-format compliance" in text
    assert "`s1-nogeneral` rejected by the gate: FAIL [3]" in text
    assert "**run1** (`checkpoints/run1`)" in text


def test_star_marks_only_the_current_candidate_and_missing_values_render_as_dashes(tmp_path):
    cfg = _cfg(tmp_path)
    rdir = results_dir(cfg)
    rdir.mkdir(parents=True)
    rows = [
        {"event": "run", "run_name": "run1", "timestamp": "1", "delta_points": 3.0, "delta_lenient_points": 1.0, "gate_passed": True, "promoted": True,
         "hparams": {"lr": None, "r": 16}, "notes": "lr 2e-4 | r=32"},
        {"event": "run", "run_name": "run2", "timestamp": "2", "delta_points": 5.0, "delta_lenient_points": 2.0, "gate_passed": True, "promoted": True,
         "train": {"step0": {"general_heldout_loss": 1.25}, "last_epoch": {"general_heldout_loss": 1.3}}},
        {"event": "run", "run_name": "cmp-only", "timestamp": "3"},
    ]
    (rdir / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (rdir / "CANDIDATE.json").write_text(json.dumps({"run_name": "run2", "model_path": "checkpoints/run2", "delta_points": 5.0}))
    text = writeup.build(cfg)
    lines = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("| run") or ln.startswith("| cmp")}
    assert lines["run2"].rstrip().endswith("| ★ |") and "promoted earlier" in lines["run1"] and "★" not in lines["run1"]
    assert "lr 2e-4 \\| r=32" in lines["run1"] and "| — | 16 |" in lines["run1"]
    assert "1.250 → 1.300" in lines["run2"] and "None" not in text and "| — |" in lines["cmp-only"]
