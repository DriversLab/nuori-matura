"""Training stack: SFTConfig mapping, loss recording, length filtering, merge helpers, pipeline wiring, sweep planning,
configs. Fast tests use fakes only; the slow test trains SmolLM2-135M for 3 LoRA steps, merges and reloads it."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eval.schema import write_jsonl
from train.config import ROOT, checkpoint_dir, checkpoints_root, get_dotted, load_config, parse_value, run_dir

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _load_script(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load_script("sweep")


def _write_cfg(tmp_path: Path, extra: dict | None = None) -> Path:
    cfg = {
        "run_name": "t",
        "model": {"base": "org/Tiny-Model"},
        "paths": {
            "results_root": str(tmp_path / "results"),
            "checkpoints_root": str(tmp_path / "ckpt"),
            "processed_root": str(tmp_path / "processed"),
        },
    }
    for key, value in (extra or {}).items():
        cur = cfg
        *parents, last = key.split(".")
        for p in parents:
            cur = cur.setdefault(p, {})
        cur[last] = value
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _row(kind: str, i: int, question: str, answer: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
        "kind": kind,
        "id": f"{kind}-{i}",
        "variant": "canonical" if kind == "matura" else None,
    }


# ----------------------------------------------------------------------------- sft: config mapping


def test_sft_kwargs_are_accepted_by_sftconfig(tmp_path):
    import torch
    from trl import SFTConfig

    from train.sft import build_sft_kwargs

    cfg = load_config(ROOT / "configs" / "run1.yaml")
    kwargs = build_sft_kwargs(cfg, output_dir=tmp_path / "trainer", device="cpu", dtype=torch.float32)
    args = SFTConfig(**kwargs)

    assert args.output_dir == str(tmp_path / "trainer")
    assert args.learning_rate == 1e-4 and args.num_train_epochs == 2.0
    assert args.warmup_steps == pytest.approx(0.03)  # ratio passed through warmup_steps (warmup_ratio removed in v5)
    assert args.lr_scheduler_type == "cosine"
    assert (args.per_device_train_batch_size, args.per_device_eval_batch_size, args.gradient_accumulation_steps) == (4, 4, 4)
    assert args.eval_strategy == "epoch" and args.save_strategy == "no"
    assert args.bf16 is False and args.fp16 is False  # never mixed precision on CPU
    assert args.gradient_checkpointing is True and args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert args.completion_only_loss is True and args.packing is False
    assert args.max_length == 1024 and args.max_steps == -1 and args.seed == 42
    assert args.dataloader_pin_memory is False
    assert kwargs["report_to"] == "none"


def test_sft_kwargs_precision_and_eval_strategy():
    import torch

    from train.sft import build_sft_kwargs

    cfg = load_config(ROOT / "configs" / "run1.yaml")
    cuda = build_sft_kwargs(cfg, output_dir="o", device="cuda", dtype=torch.bfloat16)
    assert (cuda["bf16"], cuda["fp16"], cuda["dataloader_pin_memory"]) == (True, False, True)
    mps = build_sft_kwargs(cfg, output_dir="o", device="mps", dtype=torch.float16)
    assert (mps["bf16"], mps["fp16"], mps["dataloader_pin_memory"]) == (False, True, False)
    cpu_bf16 = build_sft_kwargs(cfg, output_dir="o", device="cpu", dtype=torch.bfloat16)
    assert (cpu_bf16["bf16"], cpu_bf16["fp16"]) == (False, False)

    steps = build_sft_kwargs(load_config(ROOT / "configs" / "run1.yaml", ["train.eval_steps=50"]), output_dir="o",
                             device="cpu", dtype=torch.float32)
    assert (steps["eval_strategy"], steps["eval_steps"]) == ("steps", 50)
    assert build_sft_kwargs(cfg, output_dir="o", device="cpu", dtype=torch.float32, has_eval=False)["eval_strategy"] == "no"
    with pytest.raises(ValueError, match="warmup_ratio"):
        build_sft_kwargs(load_config(ROOT / "configs" / "run1.yaml", ["train.warmup_ratio=10"]), output_dir="o",
                         device="cpu", dtype=torch.float32)


def test_lora_config_from_cfg():
    from train.sft import build_lora_config

    lora = build_lora_config(load_config(ROOT / "configs" / "run1.yaml", ["lora.use_rslora=true"]))
    assert (lora.r, lora.lora_alpha, lora.lora_dropout, lora.bias, lora.use_rslora) == (16, 32, 0.05, "none", True)
    assert str(lora.task_type).endswith("CAUSAL_LM") and lora.target_modules == "all-linear"


# ----------------------------------------------------------------------------- sft: losses and data


def test_loss_recorder_groups_step0_and_epochs():
    from train.sft import LossRecorder

    rec = LossRecorder()
    s0 = SimpleNamespace(global_step=0, epoch=None)
    for name, value in (("matura", 2.0), ("general", 3.0), ("general_heldout", 4.0)):  # one call per eval dataset
        rec.on_evaluate(None, s0, None, metrics={f"eval_{name}_loss": value, f"eval_{name}_runtime": 0.1})
    s1 = SimpleNamespace(global_step=2, epoch=0.5)
    rec.on_log(None, s1, None, logs={"loss": 1.5, "grad_norm": 1.0})
    rec.on_log(None, s1, None, logs={"loss": 0.5})
    rec.on_log(None, s1, None, logs={"eval_matura_loss": 9.0})  # eval logs are not train losses
    s2 = SimpleNamespace(global_step=4, epoch=1.0)
    rec.on_evaluate(None, s2, None, metrics={"eval_matura_loss": 1.0})
    rec.on_evaluate(None, s2, None, metrics={"eval_general_loss": 2.5})

    assert rec.step0() == {"matura_loss": 2.0, "general_loss": 3.0, "general_heldout_loss": 4.0}
    assert rec.last_step() == 4
    assert rec.epochs() == [
        {"epoch": 1.0, "step": 4, "train_loss": 1.0, "matura_loss": 1.0, "general_loss": 2.5, "general_heldout_loss": None}
    ]


class CharTokenizer:
    """apply_chat_template stand-in: one token per content character (the generation prompt adds none)."""

    def apply_chat_template(self, conversations, return_dict=True, add_generation_prompt=False):
        assert return_dict is False
        return [[0] * sum(len(m["content"]) for m in conv) for conv in conversations]


def test_load_splits_filters_too_long_and_drops_extra_columns(tmp_path):
    from train.sft import load_splits

    data = {
        "train": tmp_path / "train.jsonl",
        "val_matura": tmp_path / "val_matura.jsonl",
        "val_general": None,
        "general_heldout": tmp_path / "general_heldout.jsonl",
    }
    write_jsonl([_row("matura", 1, "q" * 5, "a"), _row("general", 2, "q" * 30, "a"), _row("matura", 3, "q", "a" * 9),
                 _row("general", 5, "q", "g" * 8)], data["train"])
    write_jsonl([_row("matura", 4, "q" * 25, "a")], data["val_matura"])
    write_jsonl([{"id": "g1", "source": "x", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "r"}]}],
        data["general_heldout"])

    rows, dropped, composition = load_splits(data, CharTokenizer(), max_length=10)
    assert dropped == {"train": 1, "matura": 1, "general": 0, "general_heldout": 0}
    assert [len(rows[k]) for k in ("train", "matura", "general", "general_heldout")] == [3, 0, 0, 1]
    # the dropped over-long row was general: the trained ratio is what the trainer sees, and loss tokens weigh by length
    assert composition == {"dropped_too_long_by_kind": {"general": 1, "matura": 0}, "general_ratio_trained": round(1 / 3, 4),
                           "loss_tokens": {"matura": 10, "general": 8, "matura_share": round(10 / 18, 4)}}
    assert all(set(r) == {"prompt", "completion"} for r in rows["train"])
    assert rows["general_heldout"][0] == {
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "completion": [{"role": "assistant", "content": "r"}],
    }
    with pytest.raises(RuntimeError, match="no training rows"):
        load_splits(data, CharTokenizer(), max_length=1)


def test_run_name_guard_protects_base_and_candidate(tmp_path):
    from train.sft import assert_trainable_run_name

    cfg = load_config(_write_cfg(tmp_path))
    assert_trainable_run_name(cfg)
    for reserved in ("base", "base-limit5"):
        with pytest.raises(ValueError, match="reserved"):
            assert_trainable_run_name({**cfg, "run_name": reserved})

    results = tmp_path / "results" / "tiny-model"
    results.mkdir(parents=True)
    (results / "CANDIDATE.json").write_text(json.dumps({"run_name": "t"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CANDIDATE"):
        assert_trainable_run_name(cfg)

    (results / "CANDIDATE.json").write_text(json.dumps({"run_name": "other"}), encoding="utf-8")
    checkpoint_dir(cfg).mkdir(parents=True)
    (tmp_path / "ckpt" / "CANDIDATE-tiny-model").symlink_to("t")
    with pytest.raises(RuntimeError, match="CANDIDATE"):
        assert_trainable_run_name(cfg)


def test_run_name_guard_protects_other_base_models_checkpoints(tmp_path):
    """checkpoints/<run_name> is shared by every base model: an 11B sweep reusing s1-* names must not rmtree the 4.5B
    candidate (after the 11B promotion moved nothing but its own link) nor any 4.5B run's checkpoint."""
    from train import registry
    from train.sft import assert_trainable_run_name

    small = load_config(_write_cfg(tmp_path))
    big = {**load_config(_write_cfg(tmp_path)), "model": {**small["model"], "base": "org/Big-Model"}}
    ck_root = checkpoints_root(small)
    for cfg, name in ((small, "s1"), (big, "x11b")):
        (ck_root / name / "adapter").mkdir(parents=True)
        registry._install_candidate(registry.cfg_results_dir(cfg), ck_root, {
            "run_name": name, "model_path": str(ck_root / name), "delta_points": 1.0, "tags": []})
    assert (ck_root / "CANDIDATE-tiny-model").is_symlink() and (ck_root / "CANDIDATE-big-model").is_symlink()
    with pytest.raises(RuntimeError, match="CANDIDATE checkpoint"):
        assert_trainable_run_name({**big, "run_name": "s1"})

    (ck_root / "z").mkdir()
    (ck_root / "z" / "training_meta.json").write_text(json.dumps({"base_model": "org/Tiny-Model"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="trained on org/Tiny-Model"):
        assert_trainable_run_name({**big, "run_name": "z"})
    assert_trainable_run_name({**small, "run_name": "z"})  # the same base model may retrain its own run


def test_fit_seeds_lora_init_from_cfg_seed(tmp_path, monkeypatch):
    """SFTTrainer creates the LoRA layers (kaiming init from the global torch RNG) before Trainer.__init__ seeds."""
    import torch
    import trl

    from train import sft

    draws: list[torch.Tensor] = []

    class FakeTrainer:
        def __init__(self, *, model, args, train_dataset, eval_dataset, processing_class, peft_config, callbacks):
            draws.append(torch.rand(4))  # what get_peft_model's lora_A init would draw
            self.train_dataset, self.state = train_dataset, SimpleNamespace(global_step=1)

        def evaluate(self):
            return {}

        def train(self):
            return SimpleNamespace(training_loss=0.5)

        def save_model(self, path):
            pass

    monkeypatch.setattr(trl, "SFTTrainer", FakeTrainer)
    monkeypatch.setattr(sft, "load_model", lambda *a, **k: torch.nn.Linear(2, 2))
    cfg = load_config(_write_cfg(tmp_path))
    rows = {"train": [_row("matura", 1, "q", "a")], "matura": [], "general": [], "general_heldout": []}
    rows = {k: [{"prompt": r["prompt"], "completion": r["completion"]} for r in v] for k, v in rows.items()}
    for process_seed in (1, 2):  # stands in for two fresh processes with different default torch seeds
        torch.manual_seed(process_seed)
        sft._fit(cfg, rows, tokenizer=None, device="cpu", quant="none", dtype=torch.float32, ckpt=tmp_path / f"ck{process_seed}")
    assert torch.equal(draws[0], draws[1])


def test_train_metrics_count_only_training_rows_as_dropped(tmp_path, monkeypatch):
    """Eval-split drops used to be summed into dropped_too_long (runs.jsonl then reported lost training rows)."""
    from train import sft

    cfg = load_config(_write_cfg(tmp_path), ["model.max_length=10"])
    data = {k: tmp_path / f"{k}.jsonl" for k in ("train", "val_matura", "val_general", "general_heldout")}
    write_jsonl([_row("matura", 1, "q", "a"), _row("general", 2, "q", "g" * 4)], data["train"])
    write_jsonl([_row("matura", 3, "q" * 30, "a")], data["val_matura"])
    write_jsonl([_row("general", 4, "q" * 30, "a")], data["val_general"])
    write_jsonl([], data["general_heldout"])
    monkeypatch.setattr(sft, "load_tokenizer", lambda *a, **k: CharTokenizer())
    monkeypatch.setattr(sft, "_fit", lambda *a, **k: {"step0": {}, "epochs": [], "n_train_prepared": 2, "global_steps": 1,
                                                      "train_loss_mean": 0.1})
    metrics = sft.train_model(cfg, data)["train_metrics"]
    assert metrics["dropped_too_long"] == 0 and metrics["dropped_too_long_by_split"] == {"train": 0, "matura": 1, "general": 1,
                                                                                         "general_heldout": 0}
    assert metrics["general_ratio_trained"] == 0.5 and metrics["loss_tokens"] == {"matura": 1, "general": 4, "matura_share": 0.2}
    saved = json.loads((checkpoint_dir(cfg) / "train_metrics.json").read_text(encoding="utf-8"))
    assert saved["loss_tokens"] == metrics["loss_tokens"]


# ----------------------------------------------------------------------------- merge helpers


def test_merge_device_and_artifact_copy(tmp_path):
    import torch

    from train.merge import copy_run_artifacts, merge_and_export, resolve_merge_device

    assert resolve_merge_device("cpu") == "cpu"
    assert resolve_merge_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError):
        resolve_merge_device("tpu")

    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    (ckpt / "adapter").mkdir(parents=True)
    out.mkdir()
    (ckpt / "train_metrics.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter" / "resolved_config.yaml").write_text("a: 1\n", encoding="utf-8")
    (out / "training_meta.json").write_text('{"keep": true}', encoding="utf-8")
    (ckpt / "training_meta.json").write_text('{"keep": false}', encoding="utf-8")
    assert sorted(copy_run_artifacts(ckpt / "adapter", out)) == ["resolved_config.yaml", "train_metrics.json"]
    assert json.loads((out / "training_meta.json").read_text(encoding="utf-8")) == {"keep": True}

    with pytest.raises(FileNotFoundError, match="adapter"):
        merge_and_export(ckpt, tmp_path / "merged")  # no adapter_config.json
    (ckpt / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        merge_and_export(ckpt / "adapter", ckpt / "adapter")

    # a typo'd rebuild must not silently replace another run's (e.g. the candidate's) merged weights
    other = tmp_path / "other"
    (other / "adapter").mkdir(parents=True)
    (other / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to another run"):
        merge_and_export(ckpt / "adapter", other)
    (tmp_path / "CANDIDATE-x").symlink_to("ckpt")
    with pytest.raises(ValueError, match="belongs to another run"):
        merge_and_export(other / "adapter", tmp_path / "CANDIDATE-x")


# ----------------------------------------------------------------------------- pipeline wiring


@pytest.fixture
def fake_stages(monkeypatch, tmp_path):
    """Replace every sibling stage module with recording fakes; returns (cfg, calls)."""
    cfg = load_config(_write_cfg(tmp_path))
    calls: list[tuple[str, tuple, dict]] = []
    promoted = {"value": False}

    def install(module_name: str, **fns):
        module = types.ModuleType(module_name)
        for fn_name, fn in fns.items():
            def recorder(*args, _fn=fn, _name=fn_name, **kwargs):
                calls.append((_name, args, kwargs))
                return _fn(*args, **kwargs)
            setattr(module, fn_name, recorder)
        monkeypatch.setitem(sys.modules, module_name, module)

    def train_model(cfg, data):
        adapter = checkpoint_dir(cfg) / "adapter"
        adapter.mkdir(parents=True)
        return {"adapter_dir": adapter, "train_metrics": {"n_train": 3}}

    def merge_and_export(adapter_dir, out_dir, *, base_model=None):
        (Path(out_dir) / "config.json").write_text("{}", encoding="utf-8")
        return Path(out_dir)

    def ensure_candidate(cfg):
        if promoted["value"]:
            return {"run_name": "t", "model_path": str(checkpoint_dir(cfg)), "delta_points": 2.5}
        return {"run_name": "base", "model_path": cfg["model"]["base"], "delta_points": 0.0}

    def record_run(cfg, **kwargs):
        promoted["value"] = True
        return {"gate_passed": True, "gate_reasons": ["PASS [1] headline delta +2.50 pts >= min +0.50"]}

    install("train.dataset", build_processed=lambda cfg: {"train": tmp_path / "train.jsonl", "stats": {"n": 3}})
    install("train.sft", train_model=train_model, assert_trainable_run_name=lambda cfg: None)
    install("train.merge", merge_and_export=merge_and_export)
    install("train.registry", ensure_candidate=ensure_candidate, record_run=record_run)
    install("eval.runner", run_eval=lambda model_path, run_name, cfg, **kw: {"run_name": run_name}, eval_sets_from_cfg=lambda cfg: [])
    install("eval.compare", write_compare=lambda run_dir, base_dir: {"run_name": "t"}, format_compare_table=lambda cmp: "COMPARE-TABLE")
    return cfg, calls


def test_pipeline_runs_stages_in_order_with_contract_arguments(fake_stages, capsys):
    from train.pipeline import run_pipeline

    cfg, calls = fake_stages
    ckpt = checkpoint_dir(cfg)
    result = run_pipeline(cfg)

    assert [c[0] for c in calls] == [
        "ensure_candidate", "assert_trainable_run_name", "eval_sets_from_cfg", "build_processed", "train_model",
        "merge_and_export", "run_eval", "run_eval", "write_compare", "record_run", "ensure_candidate", "format_compare_table",
    ]
    by_name: dict[str, list] = {}
    for name, args, kwargs in calls:
        by_name.setdefault(name, []).append((args, kwargs))
    assert by_name["merge_and_export"] == [((ckpt / "adapter", ckpt), {"base_model": "org/Tiny-Model"})]
    (base_args, base_kw), (run_args, run_kw) = by_name["run_eval"]
    assert base_args[:2] == ("org/Tiny-Model", "base") and base_kw == {"is_base": True}
    assert run_args[:2] == (str(ckpt), "t") and run_kw == {"force": True}
    assert by_name["write_compare"] == [((run_dir(cfg), run_dir(cfg, "base")), {})]
    (_, record_kw), = by_name["record_run"]
    assert (record_kw["run_dir"], record_kw["model_path"], record_kw["adapter_path"]) == (run_dir(cfg), ckpt, ckpt / "adapter")
    assert record_kw["cmp"] == {"run_name": "t"}

    stages = result["train_metrics"]["stage_seconds"]
    assert list(stages) == ["data", "train", "merge", "eval_base", "eval_run", "compare", "refresh_candidate"]
    saved = json.loads((ckpt / "train_metrics.json").read_text(encoding="utf-8"))
    assert list(saved["stage_seconds"]) == list(stages)
    assert result["gate"]["passed"] is True and result["candidate"]["run_name"] == "t"

    out = capsys.readouterr().out
    assert "COMPARE-TABLE" in out and "Gate: PASS" in out and "PASS [1]" in out
    assert "Current candidate: t, delta_points=+2.50  (promoted by this run)" in out
    assert f"python scripts/run_eval.py --config {cfg['_config_path']} --model {ckpt}" in out


def test_pipeline_without_eval_and_flag_rules(fake_stages):
    from train.pipeline import run_pipeline

    cfg, calls = fake_stages
    result = run_pipeline(cfg, do_eval=False)
    names = [c[0] for c in calls]
    assert "merge_and_export" in names and "run_eval" not in names and "record_run" not in names
    assert "eval_sets_from_cfg" not in names, "eval sets are only preflighted when the run will be evaluated"
    assert result["model_dir"] == checkpoint_dir(cfg) and "compare" not in result

    with pytest.raises(ValueError, match="merged export"):
        run_pipeline(cfg, do_merge=False, do_eval=True)
    calls.clear()
    no_auto = {**cfg, "pipeline": {**cfg["pipeline"], "auto_eval": False}, "run_name": "t2"}
    run_pipeline(no_auto)
    assert "run_eval" not in [c[0] for c in calls]


def test_pipeline_re_measures_a_candidate_from_another_eval_fingerprint(fake_stages, capsys, monkeypatch):
    """After an eval change the candidate's stored delta is on another scale: re-evaluate + re-record it before gating."""
    from train import pipeline

    cfg, calls = fake_stages
    stale_ck = checkpoints_root(cfg) / "run0"
    stale_ck.mkdir(parents=True)
    registry = sys.modules["train.registry"]
    candidate = {"run_name": "run0", "model_path": str(stale_ck), "adapter_path": None, "delta_points": 3.0, "fingerprint": "fp-old"}
    monkeypatch.setattr(registry, "ensure_candidate", lambda cfg: calls.append(("ensure_candidate", (), {})) or candidate)
    monkeypatch.setattr(sys.modules["eval.compare"], "write_compare",
                        lambda run_dir, base_dir: calls.append(("write_compare", (run_dir, base_dir), {})) or {"run_name": Path(run_dir).name, "fingerprint": "fp-new"})

    pipeline.refresh_stale_candidate(cfg, "fp-new")
    names = [(c[0], c[1][:2]) for c in calls]
    assert names == [("ensure_candidate", ()), ("run_eval", (str(stale_ck), "run0")),
                     ("write_compare", (run_dir(cfg, "run0"), run_dir(cfg, "base"))), ("record_run", (cfg,))]
    assert calls[-1][2]["model_path"] == stale_ck and calls[-1][2]["run_dir"] == run_dir(cfg, "run0")

    calls.clear()
    assert pipeline.refresh_stale_candidate(cfg, "fp-old") is None and [c[0] for c in calls] == ["ensure_candidate"]
    candidate["model_path"] = str(checkpoints_root(cfg) / "elsewhere")
    calls.clear()
    assert pipeline.refresh_stale_candidate(cfg, "fp-new") is None and "not on this machine" in capsys.readouterr().out
    assert [c[0] for c in calls] == ["ensure_candidate"]


def test_eval_vram_preflight_refuses_bf16_eval_that_cannot_fit(tmp_path, monkeypatch):
    import torch

    import eval.modeling as modeling
    from train import pipeline

    local = tmp_path / "local-model"
    local.mkdir()
    (local / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 123}}), encoding="utf-8")
    assert pipeline._bf16_weight_bytes(str(local)) == 123

    cfg = load_config(_write_cfg(tmp_path))  # 11B bf16 (20.8 GiB of weights) on a 22.5 GiB L4
    monkeypatch.setattr(pipeline, "_bf16_weight_bytes", lambda base: int(20.8 * 2**30))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(22 * 2**30), int(22.5 * 2**30)))
    monkeypatch.setattr(modeling, "detect_device", lambda: "mps")
    pipeline.check_eval_vram(cfg)  # not CUDA: nothing to check
    monkeypatch.setattr(modeling, "detect_device", lambda: "cuda")
    with pytest.raises(RuntimeError, match="eval.quantization=4bit"):
        pipeline.check_eval_vram(cfg)
    pipeline.check_eval_vram({**cfg, "eval": {**cfg["eval"], "quantization": "4bit"}})
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(79 * 2**30), int(80 * 2**30)))
    pipeline.check_eval_vram(cfg)


def test_candidate_commands_for_base_and_run(tmp_path):
    from train.pipeline import candidate_commands, format_verdict

    cfg = load_config(_write_cfg(tmp_path))
    base = candidate_commands(cfg, {"run_name": "base", "model_path": "org/Tiny-Model"})
    assert base[1].endswith("scripts/run_eval.py --config " + cfg["_config_path"] + " --base --model org/Tiny-Model")
    run = candidate_commands(cfg, {"run_name": "r1", "model_path": str(tmp_path / "ckpt" / "r1")})
    assert f"--model {tmp_path / 'ckpt' / 'r1'}" in run[1] and "stable link" not in run[0]
    verdict = format_verdict(cfg, run_name="r1", passed=True, reasons=[], candidate={
        "run_name": "r1", "model_path": str(tmp_path / "ckpt" / "r1"), "delta_points": 2.0, "warnings": ["no merged model at ckpt/r1"]})
    assert "  WARNING: no merged model at ckpt/r1" in verdict.splitlines()

    link = checkpoints_root(cfg) / "CANDIDATE-tiny-model"
    (checkpoints_root(cfg) / "r1").mkdir(parents=True)
    link.symlink_to("r1")
    linked = candidate_commands(cfg, {"run_name": "r1", "model_path": str(checkpoints_root(cfg) / "r1")})
    assert linked[0].endswith(f"(stable link: {link})"), "the link itself, not its resolved target"
    (checkpoints_root(cfg) / "r2").mkdir()
    other = candidate_commands(cfg, {"run_name": "r2", "model_path": str(checkpoints_root(cfg) / "r2")})
    assert "stable link" not in other[0], "a link that points at another run is never shown"
    (checkpoints_root(cfg) / "CANDIDATE-big-model").symlink_to("r2")
    assert "stable link" not in candidate_commands(cfg, {"run_name": "r2", "model_path": str(checkpoints_root(cfg) / "r2")})[0], \
        "another base model's link is never shown"


# ----------------------------------------------------------------------------- sweep


def test_grid_expansion_and_names(sweep):
    runs = sweep.expand_grid({"train.learning_rate": [1.0e-4, 2.0e-4], "lora.r": [16, 32]})
    assert [r["name"] for r in runs] == [
        "g-learning_rate=0.0001_r=16", "g-learning_rate=0.0001_r=32", "g-learning_rate=0.0002_r=16", "g-learning_rate=0.0002_r=32",
    ]
    assert runs[1]["set"] == {"train.learning_rate": 1.0e-4, "lora.r": 32}
    clash = sweep.expand_grid({"train.x": [1], "eval.x": [2]})
    assert clash[0]["name"] == "g-train.x=1_eval.x=2"
    assert sweep.expand_grid({}) == []


@pytest.mark.parametrize("value", [2.0e-4, 1e-05, 32, 0.0, True, None, "cosine", "a: b", [16, 32],
                                   {"canonical": 0.45, "reason_then_answer": 0.55}, "g-learning_rate=0.0002_r=16"])
def test_override_values_round_trip_through_config_parser(sweep, value):
    assert parse_value(sweep.format_value(value)) == value


def _sweep_spec(tmp_path: Path, **extra) -> dict:
    return {"name": "sw", "base_config": str(_write_cfg(tmp_path)), "budget_hours": 5, **extra}


def test_plan_runs_commands_logs_and_validation(sweep, tmp_path):
    spec = _sweep_spec(tmp_path, runs=[{"name": "a", "set": {"train.learning_rate": 2.0e-4,
                                                             "data.variant_mix": {"canonical": 1.0}}}],
                       grid={"lora.r": [16, 32]})
    runs = sweep.plan_runs(spec)
    assert [r.name for r in runs] == ["a", "g-r=16", "g-r=32"]
    cmd = runs[0].command
    assert cmd[:6] == [sys.executable, "scripts/train.py", "--config", spec["base_config"], "--set", "run_name=a"]
    assert runs[0].log_path == tmp_path / "results" / "tiny-model" / "sweeps" / "sw" / "a.log"

    train_script = _load_script("train")
    args = train_script.parse_args(cmd[2:])
    cfg = load_config(args.config, args.set)
    assert cfg["run_name"] == "a" and cfg["tags"] == ["sw"] and cfg["train"]["learning_rate"] == 2.0e-4
    assert cfg["data"]["variant_mix"] == {"canonical": 1.0} and cfg["notes"].startswith("sweep sw: ")

    with pytest.raises(ValueError, match="unknown config keys"):
        sweep.plan_runs(_sweep_spec(tmp_path, runs=[{"name": "x", "set": {"train.learnig_rate": 1.0}}]))
    with pytest.raises(ValueError, match="duplicate"):
        sweep.plan_runs(_sweep_spec(tmp_path, runs=[{"name": "g-r=16", "set": {}}], grid={"lora.r": [16]}))


def _append_row(results: Path, row: dict) -> None:
    results.mkdir(parents=True, exist_ok=True)
    with open(results / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "run", **row}) + "\n")


def _run_name(command: list[str]) -> str:
    return next(a.split("=", 1)[1] for a in command if a.startswith("run_name="))


def test_run_sweep_skips_recorded_continues_after_failure_and_stops_on_budget(sweep, tmp_path):
    spec = _sweep_spec(tmp_path, runs=[{"name": n, "set": {}} for n in "abcde"])
    runs = sweep.plan_runs(spec)
    _append_row(tmp_path / "results" / "tiny-model", {"run_name": "a", "delta_points": 1.0})
    now = [0.0]
    launched: list[str] = []

    def runner(command, log_path):
        launched.append(_run_name(command))
        now[0] += 2 * 3600
        return 1 if launched[-1] == "b" else 0

    records = sweep.run_sweep(runs, budget_hours=5, est_hours=1.0, runner=runner, clock=lambda: now[0])
    # b (fails, est stays 1h) at t=0, c at t=2h (5-2=3h >= 1h), then est=2h from c and 1h left -> stop
    assert launched == ["b", "c"]
    status = {r["name"]: r["status"] for r in records}
    assert status["a"] == "skipped" and status["b"] == "failed" and status["c"] == "ok"
    assert status["d"].startswith("not-run (budget") and status["e"].startswith("not-run (budget")

    launched.clear()
    capped = sweep.run_sweep(runs, budget_hours=100, est_hours=1.0, max_runs=1, runner=runner, clock=lambda: now[0])
    assert launched == ["b"] and capped[2]["status"] == "not-run (max-runs 1 reached)"


def test_run_sweep_without_estimate_launches_until_measured_runs_no_longer_fit(sweep, tmp_path):
    runs = sweep.plan_runs(_sweep_spec(tmp_path, runs=[{"name": n, "set": {}} for n in "abc"]))
    now = [0.0]
    launched: list[str] = []

    def runner(command, log_path):
        launched.append(_run_name(command))
        now[0] += 0.4 * 3600
        return 0

    records = sweep.run_sweep(runs, budget_hours=1, est_hours=None, runner=runner, clock=lambda: now[0])
    # a at t=0 (nothing measured, budget left), b at 0.4h (0.6h left >= 0.4h), c: 0.2h left < 0.4h
    assert launched == ["a", "b"] and records[2]["status"].startswith("not-run (budget")


def test_sweep_main_dry_run_and_summary(sweep, tmp_path, capsys):
    results = tmp_path / "results" / "tiny-model"
    spec = _sweep_spec(tmp_path, runs=[{"name": "a", "set": {}}, {"name": "b", "set": {"lora.r": 32}}, {"name": "c", "set": {}}])
    sweep_file = tmp_path / "sweep.yaml"
    sweep_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    _append_row(results, {"run_name": "a", "delta_points": 1.0, "gate_passed": True, "gate_failed_rules": []})

    def never(command, log_path):
        raise AssertionError("dry run must not launch")

    assert sweep.main(["--sweep", str(sweep_file), "--dry-run"], runner=never) == 0
    out = capsys.readouterr().out
    assert "# a: skip" in out and "run_name=b" in out and "lora.r=32" in out and "2 to launch" in out

    def runner(command, log_path):
        name = _run_name(command)
        if name == "c":
            return 1
        _append_row(results, {"run_name": name, "delta_points": 3.0, "gate_passed": True, "gate_failed_rules": [],
                              "ci95": [0.5, 5.5], "general_nll_rel_change": 0.01})
        (results / "CANDIDATE.json").write_text(json.dumps({"run_name": name, "delta_points": 3.0, "model_path": "ck/b"}))
        return 0

    assert sweep.main(["--sweep", str(sweep_file), "--budget-hours", "10"], runner=runner) == 1
    out = capsys.readouterr().out
    assert "Current candidate" in out and "b (+3.00 pts)" in out
    summary = json.loads((results / "sweeps" / "sw" / "sweep_summary.json").read_text(encoding="utf-8"))
    assert [r["name"] for r in summary["ranking"]] == ["b", "a", "c"]
    assert [r["status"] for r in summary["runs"]] == ["skipped", "ok", "failed"]
    assert summary["candidates"][str(results)]["run_name"] == "b"


def test_sweep_refuses_a_recorded_run_name_whose_config_changed(sweep, tmp_path):
    spec = _sweep_spec(tmp_path, runs=[{"name": "s1-lr", "set": {"train.learning_rate": 2.0e-4}}])
    (planned,) = sweep.plan_runs(spec)
    results = tmp_path / "results" / "tiny-model"
    train_script = _load_script("train")
    args = train_script.parse_args(planned.command[2:])
    assert planned.config_hash == train_script_hash(args), "the hash record_run stores for this command"
    _append_row(results, {"run_name": "s1-lr", "delta_points": 2.5, "config_hash": planned.config_hash})
    assert sweep.recorded_status(planned) == "same"
    sweep.check_changed_runs([planned])

    (edited,) = sweep.plan_runs(_sweep_spec(tmp_path, runs=[{"name": "s1-lr", "set": {"train.learning_rate": 3.0e-4}}]))
    assert sweep.recorded_status(edited) == "changed"
    sweep_file = tmp_path / "sweep.yaml"
    sweep_file.write_text(yaml.safe_dump(_sweep_spec(tmp_path, runs=[{"name": "s1-lr", "set": {"train.learning_rate": 3.0e-4}}])))
    with pytest.raises(ValueError, match="different config"):
        sweep.main(["--sweep", str(sweep_file), "--dry-run"])
    _append_row(results, {"run_name": "legacy", "delta_points": 1.0})
    (legacy,) = sweep.plan_runs(_sweep_spec(tmp_path, runs=[{"name": "legacy", "set": {}}]))
    assert sweep.recorded_status(legacy) == "same", "rows without a hash still resume"


def train_script_hash(args) -> str:
    from train.config import config_hash

    return config_hash(load_config(args.config, args.set))


def test_config_typo_guard_and_atomic_variant_mix(tmp_path):
    with pytest.raises(ValueError, match="did you mean 'train.learning_rate'"):
        load_config(ROOT / "configs" / "run1.yaml", ["train.learnig_rate=2e-4"])
    for bad in ("lora.rank=64", "eval.limt=5"):
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(ROOT / "configs" / "run1.yaml", [bad])
    section = tmp_path / "typo.yaml"
    section.write_text("training:\n  epochs: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key 'training.epochs'"):
        load_config(section)
    ok = load_config(ROOT / "configs" / "run1.yaml", ["run_name=x", "tags=[a]", "data.variant_mix.canonical=1.0"])
    assert ok["run_name"] == "x" and ok["data"]["variant_mix"]["canonical"] == 1.0

    only_canonical = tmp_path / "vm.yaml"
    only_canonical.write_text("data:\n  variant_mix: {canonical: 1.0}\n", encoding="utf-8")
    from_file = load_config(only_canonical)["data"]["variant_mix"]
    assert from_file == {"canonical": 1.0} == load_config(None, ["data.variant_mix={canonical: 1.0}"])["data"]["variant_mix"]
    assert load_config(only_canonical)["data"]["general_ratio"] == 0.25, "other data keys still merge"


# ----------------------------------------------------------------------------- configs and CLIs


def test_owned_configs_resolve(sweep):
    run1 = load_config(ROOT / "configs" / "run1.yaml")
    assert (run1["run_name"], run1["tags"], run1["train"]["learning_rate"], run1["lora"]["r"], run1["lora"]["alpha"],
            run1["train"]["epochs"], run1["data"]["general_ratio"]) == ("run1", ["v0-safe"], 1.0e-4, 16, 32, 2, 0.3)
    assert run1["notes"]

    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert smoke["model"]["base"] == TINY and smoke["model"]["quantization"] == "none"
    assert smoke["gate"]["min_delta_points"] == -1000 and smoke["train"]["max_steps"] == 6
    assert all(str(get_dotted(smoke, f"paths.{k}")).startswith(".smoke/") for k in ("results_root", "checkpoints_root", "processed_root"))

    big = load_config(ROOT / "configs" / "bielik11b.yaml")
    assert (big["run_name"], big["model"]["base"], big["model"]["quantization"]) == ("bielik11b-v0", "speakleash/Bielik-11B-v2.6-Instruct", "auto")
    assert (big["train"]["per_device_batch_size"], big["train"]["grad_accum"], big["eval"]["batch_size"]) == (2, 8, 8)

    spec = sweep.load_sweep(ROOT / "configs" / "sweep.yaml")
    runs = sweep.plan_runs(spec)  # validates every override key against run1
    assert spec["name"] == "sweep1" and len(runs) == 6 and len({r.name for r in runs}) == 6


@pytest.mark.parametrize("script", ["train.py", "merge_and_export.py", "sweep.py"])
def test_script_help(script):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "usage:" in proc.stdout


# ----------------------------------------------------------------------------- slow: real tiny LoRA run


@pytest.mark.slow
def test_tiny_lora_train_merge_reload(tmp_path):
    import torch
    from safetensors.torch import load_file

    from eval.modeling import free_model, load_model, load_tokenizer
    from train.merge import merge_and_export
    from train.sft import train_model

    matura = [_row("matura", i, f"Ile to {i} + {i}?\nA. {2 * i}\nB. {2 * i + 1}\n\nOdpowiedz w formacie 'Odpowiedź: X'.",
                   "Odpowiedź: A") for i in range(8)]
    general = [_row("general", i, f"Napisz jedno zdanie o liczbie {i}.", f"Liczba {i} jest liczbą całkowitą.") for i in range(4)]
    too_long = _row("matura", 99, "słowo " * 600, "Odpowiedź: B")
    data = {k: tmp_path / "data" / f"{k}.jsonl" for k in ("train", "val_matura", "val_general", "general_heldout")}
    write_jsonl(matura[:6] + general[:2] + [too_long], data["train"])
    write_jsonl(matura[6:], data["val_matura"])
    write_jsonl(general[2:3], data["val_general"])
    write_jsonl([{"id": "h1", "source": "test", "messages": general[3]["prompt"] + general[3]["completion"]}], data["general_heldout"])
    data["stats"] = {"n_train_rows": 9}

    cfg = load_config(ROOT / "configs" / "smoke.yaml", {
        "run_name": "tiny", "train.max_steps": 3, "train.learning_rate": 1.0e-3,
        "paths.results_root": str(tmp_path / "results"), "paths.checkpoints_root": str(tmp_path / "ckpt"),
    })
    out = train_model(cfg, data)
    ckpt, adapter, metrics = tmp_path / "ckpt" / "tiny", out["adapter_dir"], out["train_metrics"]

    assert adapter == ckpt / "adapter"
    assert (adapter / "adapter_config.json").exists() and (adapter / "adapter_model.safetensors").exists()
    assert metrics["n_train"] == 8 and metrics["dropped_too_long"] == 1 and metrics["dropped_too_long_by_split"]["train"] == 1
    assert all(math.isfinite(metrics["step0"][k]) for k in ("matura_loss", "general_loss", "general_heldout_loss"))
    last = metrics["epochs"][-1]
    assert last["step"] == 3 and math.isfinite(last["train_loss"]) and math.isfinite(last["matura_loss"])
    assert last["matura_loss"] != metrics["step0"]["matura_loss"]
    lora_b = [t for k, t in load_file(adapter / "adapter_model.safetensors").items() if "lora_B" in k]
    assert lora_b and any(float(t.abs().sum()) > 0 for t in lora_b)  # B starts at zero: it trained
    meta = json.loads((ckpt / "training_meta.json").read_text(encoding="utf-8"))
    assert meta["base_model"] == TINY and meta["run_name"] == "tiny" and meta["config_hash"]
    assert json.loads((ckpt / "train_metrics.json").read_text(encoding="utf-8"))["step0"] == metrics["step0"]
    assert (ckpt / "resolved_config.yaml").exists()

    merged = merge_and_export(adapter, tmp_path / "merged")
    assert not (merged / "adapter_config.json").exists()
    assert (merged / "config.json").exists() and (merged / "tokenizer_config.json").exists()
    assert all((merged / f).exists() for f in ("train_metrics.json", "resolved_config.yaml", "training_meta.json"))
    assert not any("lora_" in k for k in load_file(merged / "model.safetensors"))

    model = load_model(merged, dtype="fp32", device="cpu")
    tok = load_tokenizer(merged)
    assert model._matura_load_info["adapter"] is None
    enc = tok.apply_chat_template([matura[0]["prompt"][0]], add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        first = model.generate(**enc, max_new_tokens=8, do_sample=False)
        second = model.generate(**enc, max_new_tokens=8, do_sample=False)
    del model
    free_model(None)
    assert torch.equal(first, second) and first.shape[1] > enc["input_ids"].shape[1]
