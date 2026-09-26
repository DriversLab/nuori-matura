"""Evaluation orchestration: eval sets x prompt variants -> graded predictions, loglik, organizer-protocol letter
accuracy, general NLL -> summary.json.

A summary is reused only when its eval fingerprint, model identity and model weights signature all match, so
base and fine-tuned runs are compared under identical conditions and a retrained checkpoint is never served stale.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch
import yaml

from eval import general_loss, generation, loglik, modeling, organizers
from eval.answer_format import DEFAULT_VARIANT, FORMAT_VERSION, VARIANTS
from eval.grader import grade, summarize
from eval.prompts import build_messages
from eval.schema import load_items, read_jsonl, write_jsonl
from train.config import ROOT, deep_merge, get_dotted, is_base_run_name, resolve_path, run_dir

SUMMARY_FILE = "summary.json"
EVAL_LOCK = ".eval.lock"
REASONING_VARIANT = "reason_then_answer"
HEADLINE_KEYS = ("points", "max_points", "score_pct", "parse_fail_rate", "points_lenient")
# per-run artifacts invalidated by a fresh eval (compare.json is rebuilt against the new summary)
_RUN_ARTIFACTS = (SUMMARY_FILE, "compare.json", "predictions.*.jsonl", "loglik.*.jsonl", "organizer.*.jsonl")
# modules that turn a model into points (parsers, partial credit, prompt rendering, tokenization): part of the fingerprint
EVAL_CODE_FILES = ("answer_format.py", "grader.py", "prompts.py", "schema.py", "generation.py", "loglik.py", "general_loss.py")
# hashed in on top of those ONLY for evals that score a set under the organizers' protocol, so enabling `organizer`
# (or editing their prompt) invalidates cached evals while every other eval keeps its fingerprint unchanged
ORGANIZER_CODE_FILES = ("organizers.py",)
# settings a pipeline eval depends on, reproduced from a checkpoint's resolved_config.yaml (checkpoint_eval_overrides)
EVAL_CONFIG_KEYS = ("eval", "model.dtype", "model.attn_implementation", "model.max_length", "data.general_heldout")


# ----------------------------------------------------------------------------- config / identity


def _set_specs(cfg: dict) -> list[tuple[str, dict]]:
    """[(name, raw spec dict)] of eval.sets, in order (a bare string entry is its path)."""
    out = []
    for entry in get_dotted(cfg, "eval.sets") or []:
        spec = {"path": entry} if isinstance(entry, str) else dict(entry)
        out.append((spec.get("name") or Path(str(spec["path"])).stem, spec))
    return out


def organizer_sets_from_cfg(cfg: dict) -> dict[str, dict]:
    """{eval set name: {"organizer", "organizer_rotate"}} normalized (both default False) from eval.sets.

    Kept out of eval_sets_from_cfg's rows so the normalized shape of a set stays exactly {name, path, variants,
    loglik, optional}; the per-set evaluation and the eval fingerprint read the organizer flags from here.
    """
    specs = {}
    for name, spec in _set_specs(cfg):
        organizer = bool(spec.get("organizer", False))
        specs[name] = {"organizer": organizer, "organizer_rotate": organizer and bool(spec.get("organizer_rotate", False))}
    return specs


def _organizer_spec(organizer: dict[str, dict], name: str) -> dict:
    return organizer.get(name) or {"organizer": False, "organizer_rotate": False}


def eval_sets_from_cfg(cfg: dict) -> list[dict]:
    """Normalized [{name, path, variants, loglik, optional}]; optional sets whose file is missing are skipped.

    An explicit empty `variants: []` is kept: such a set is scored only under the organizers' protocol
    (organizer_sets_from_cfg), with no generation and no grading.
    """
    default_variant = get_dotted(cfg, "eval.primary_variant") or DEFAULT_VARIANT
    sets: list[dict] = []
    names: set[str] = set()
    for name, spec in _set_specs(cfg):
        path = str(spec["path"])
        variants = spec.get("variants") if spec.get("variants") is not None else [default_variant]
        variants = list(dict.fromkeys([variants] if isinstance(variants, str) else variants))
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"eval set {name!r}: unknown variants {unknown}; known: {sorted(VARIANTS)}")
        if name in names:
            raise ValueError(f"duplicate eval set name {name!r}; give one of them an explicit name")
        names.add(name)
        optional = bool(spec.get("optional", False))
        if not resolve_path(path).exists():
            if optional:
                warnings.warn(f"optional eval set {name!r} skipped: {path} not found", stacklevel=2)
                continue
            raise FileNotFoundError(f"eval set {name!r}: {path} not found")
        sets.append({"name": name, "path": path, "variants": variants, "loglik": bool(spec.get("loglik", False)), "optional": optional})
    return sets


def eval_quantization(cfg: dict) -> str:
    """Eval loads unquantized unless eval.quantization says otherwise (e.g. 4bit to fit an 11B on a small GPU)."""
    return get_dotted(cfg, "eval.quantization") or "none"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(resolve_path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info(device: str) -> dict:
    """Library versions and accelerator model: attention/matmul kernels (and so bf16 greedy outputs and loglik) differ
    across them, so evals made under different ones are not comparable."""
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        transformers_version = None
    accelerator = torch.cuda.get_device_name() if device == "cuda" else device
    return {"torch": str(torch.__version__), "transformers": transformers_version, "accelerator": accelerator}


def planned_load_info(cfg: dict) -> dict:
    """device/dtype/quantization/attention backend + environment that run_eval will load with (known before loading,
    for the cache check)."""
    device = modeling.detect_device()
    return {
        "device": device,
        "dtype": modeling.dtype_name(modeling.resolve_dtype(get_dotted(cfg, "model.dtype", "auto"), device)),
        "quantization": modeling.resolve_quantization(eval_quantization(cfg), device),
        "attn_implementation": modeling.resolve_attn_implementation(get_dotted(cfg, "model.attn_implementation", "auto")),
        **environment_info(device),
    }


def model_load_info(model) -> dict:
    """planned_load_info's keys for an already loaded model (what it was actually loaded with)."""
    info = getattr(model, "_matura_load_info", None) or {}
    device = info.get("device") or modeling.model_device(model).type
    attn = info.get("attn_implementation") or getattr(getattr(model, "config", None), "_attn_implementation", None)
    return {
        "device": device,
        "dtype": info.get("dtype") or modeling.dtype_name(model.dtype),
        "quantization": info.get("quantization") or ("4bit" if getattr(model, "is_loaded_in_4bit", False) else "none"),
        "attn_implementation": attn if isinstance(attn, str) else "unknown",
        **environment_info(device),
    }


def eval_code_sha256(extra_files: tuple[str, ...] = ()) -> str:
    """sha256-16 over EVAL_CODE_FILES (+ extra_files, e.g. ORGANIZER_CODE_FILES when a set uses that protocol;
    CRLF-normalized bytes, not an AST dump, so laptop and GPU machine agree): a cached eval graded or prompted by
    different code is never reused or compared as matching."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in (*EVAL_CODE_FILES, *extra_files):
        h.update(name.encode() + b"\0" + (here / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def _set_fingerprint_part(s: dict, organizer: dict) -> dict:
    """Fingerprint part of one eval set. Organizer keys appear only for sets that enable it, so a set without it
    keeps its fingerprint byte-for-byte."""
    part = {"name": s["name"], "sha256": file_sha256(s["path"]), "variants": s["variants"], "loglik": s["loglik"]}
    if organizer["organizer"]:
        part["organizer"] = True
        part["organizer_rotate"] = organizer["organizer_rotate"]
        part["organizer_prompt_sha256"] = organizers.organizer_prompt_sha256()
    return part


def _fingerprint(cfg: dict, sets: list[dict], load_info: dict) -> tuple[str, dict]:
    ev = cfg.get("eval") or {}
    organizer = organizer_sets_from_cfg(cfg)
    set_parts = [_set_fingerprint_part(s, _organizer_spec(organizer, s["name"])) for s in sets]
    uses_organizer = any(p.get("organizer") for p in set_parts)
    parts: dict = {
        "format_version": FORMAT_VERSION,
        # plain call when nothing uses the organizer protocol: such an eval keeps its fingerprint unchanged
        "eval_code_sha256": eval_code_sha256(ORGANIZER_CODE_FILES) if uses_organizer else eval_code_sha256(),
        "sets": set_parts,
        "max_new_tokens": ev.get("max_new_tokens"),
        "system_prompt": ev.get("system_prompt"),
        "limit": ev.get("limit"),
        "batch_size": int(ev.get("batch_size") or 8),
        "general_loss": None,
        **{k: load_info.get(k) for k in ("dtype", "quantization", "device", "attn_implementation", "torch", "transformers", "accelerator")},
    }
    if any(REASONING_VARIANT in s["variants"] for s in sets):
        parts["max_new_tokens_reasoning"] = ev.get("max_new_tokens_reasoning")
    if any(s["loglik"] for s in sets):
        parts["loglik_template_sha256"] = hashlib.sha256(loglik.LLMZSZL_TEMPLATE.encode()).hexdigest()[:16]
    if ev.get("general_loss"):
        heldout = get_dotted(cfg, "data.general_heldout")
        parts["general_loss"] = {
            "sha256": file_sha256(heldout) if heldout and resolve_path(heldout).exists() else None,
            "max_items": ev.get("general_loss_max_items"),
            "max_length": get_dotted(cfg, "model.max_length"),
        }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return digest, parts


def eval_fingerprint(cfg: dict, load_info: dict) -> tuple[str, dict]:
    """(sha256-16, parts) over everything that changes eval numbers except the model itself."""
    return _fingerprint(cfg, eval_sets_from_cfg(cfg), load_info)


def model_identity(model_path: str | Path) -> str:
    """Hub ids as given; local paths resolved (symlinks such as checkpoints/CANDIDATE followed), repo-relative inside the repo."""
    p = Path(model_path).expanduser()
    if not p.exists():
        return str(model_path)
    p = p.resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def model_signature(model_path: str | Path) -> str:
    """Changes whenever the weights behind model_path change (e.g. a run retrained under the same name, or merged
    weights rebuilt from another adapter / base / dtype).

    Hub ids: the id. Local dirs: content hash of the LoRA adapter files when present, plus name/size/mtime of the
    merged weight files at the dir root when present (so a checkpoint whose merged weights were cleaned up or
    rebuilt is re-evaluated).
    """
    p = Path(model_path).expanduser()
    if not p.is_dir():
        return str(model_path)
    adapter = p if modeling.is_adapter_dir(p) else p / "adapter"
    h = hashlib.sha256()
    if modeling.is_adapter_dir(adapter):
        for f in sorted(adapter.glob("adapter_*")):
            h.update(f"{f.name}:{file_sha256(f)}".encode())
    if not modeling.is_adapter_dir(p):
        for f in sorted([*p.glob("*.safetensors"), *p.glob("*.bin")]):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def resolve_model_source(model_path: str | Path) -> str:
    """Checkpoint dirs whose merged weights were cleaned up are evaluated from their adapter/ subdir."""
    p = Path(model_path).expanduser()
    if not p.is_dir() or modeling.is_adapter_dir(p) or not modeling.is_adapter_dir(p / "adapter"):
        return str(model_path)
    if any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin")):
        return str(model_path)
    print(f"notice: {p} has no merged weights; evaluating base + {p / 'adapter'} (adapter merged in memory)")
    return str(p / "adapter")


# ----------------------------------------------------------------------------- summary io


def load_summary(run_dir: str | Path) -> dict | None:
    path = Path(run_dir) / SUMMARY_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_jsonl(rows, tmp)
    os.replace(tmp, path)


@contextmanager
def _eval_lock(out_dir: Path) -> Iterator[None]:
    """One evaluator per run dir across processes: a concurrent eval of the same run (e.g. two pipelines evaluating
    the base) waits, then reuses the fresh summary instead of clearing artifacts the other process just wrote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / EVAL_LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _clear_run_artifacts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in _RUN_ARTIFACTS:
        for f in out_dir.glob(pattern):
            f.unlink()


def _primary(cfg: dict, sets: list[dict]) -> tuple[str | None, str | None]:
    """(set, variant) the headline is taken from, or (None, None) when no set has prompt variants (an
    organizer-only eval: nothing is generated or graded, so the summary has no headline)."""
    gradable = [s for s in sets if s["variants"]]
    if not gradable:
        warnings.warn("no eval set has prompt variants: the summary has no headline (organizer-only eval)", stacklevel=3)
        return None, None
    by_name = {s["name"]: s for s in gradable}
    want_set, want_variant = get_dotted(cfg, "eval.primary_set"), get_dotted(cfg, "eval.primary_variant")
    chosen = by_name.get(want_set) or gradable[0]
    variant = want_variant if want_variant in chosen["variants"] else chosen["variants"][0]
    if (chosen["name"], variant) != (want_set, want_variant):
        warnings.warn(
            f"primary {want_set}/{want_variant} is not evaluated; headline uses {chosen['name']}/{variant}", stacklevel=3
        )
    return chosen["name"], variant


# ----------------------------------------------------------------------------- evaluation


def _eval_sets(model, tokenizer, cfg: dict, sets: list[dict], out_dir: Path, run_name: str) -> dict:
    ev = cfg["eval"]
    batch_size = int(ev.get("batch_size") or 8)
    limit = ev.get("limit")
    organizer = organizer_sets_from_cfg(cfg)
    results: dict[str, dict] = {}
    for s in sets:
        items = load_items(resolve_path(s["path"]))
        if limit:
            items = items[:limit]
        entry: dict = {"path": s["path"], "n_items": len(items), "variants": {}, "loglik": None}
        for variant in s["variants"]:
            started = time.time()
            max_new = ev["max_new_tokens_reasoning"] if variant == REASONING_VARIANT else ev["max_new_tokens"]
            messages = [build_messages(item, variant, ev.get("system_prompt")) for item in items]
            outputs = generation.generate_responses(
                model, tokenizer, messages, max_new_tokens=int(max_new), batch_size=batch_size, desc=f"{s['name']}/{variant}"
            )
            records = [
                {**grade(item, output, variant), "prompt": generation.render_prompt(tokenizer, msgs)}
                for item, output, msgs in zip(items, outputs, messages)
            ]
            _write_jsonl_atomic(records, out_dir / f"predictions.{s['name']}.{variant}.jsonl")
            summary = entry["variants"][variant] = summarize(records)
            print(
                f"[{run_name}] {s['name']}/{variant}: {summary['points']:g}/{summary['max_points']:g} pts "
                f"({summary['score_pct']:.2f}%), parse-fail {100 * summary['parse_fail_rate']:.1f}% "
                f"[{time.time() - started:.0f}s]"
            )
        if s["loglik"] and any(item["type"] == "mc" for item in items):
            records = loglik.score_mc_loglik(model, tokenizer, items, batch_size=batch_size)
            _write_jsonl_atomic(records, out_dir / f"loglik.{s['name']}.jsonl")
            entry["loglik"] = loglik.summarize_loglik(records)
            print(f"[{run_name}] {s['name']} loglik acc {100 * entry['loglik']['acc']:.1f}% (n={entry['loglik']['n']})")
        org = _organizer_spec(organizer, s["name"])
        if org["organizer"]:
            entry["organizer"] = _eval_organizer(model, tokenizer, items, org, out_dir, s["name"], run_name, batch_size)
        results[s["name"]] = entry
    return results


def _eval_organizer(model, tokenizer, items, org, out_dir: Path, set_name: str, run_name: str, batch_size: int) -> dict:
    """Score one set the organizers' way (no generation, no grading) -> organizer.<set>.jsonl + summary block."""
    started = time.time()
    records = organizers.score_organizer(model, tokenizer, items, batch_size=batch_size)
    rotated = organizers.score_organizer(model, tokenizer, items, batch_size=batch_size, rotate=1) if org["organizer_rotate"] else None
    _write_jsonl_atomic(records + (rotated or []), out_dir / f"organizer.{set_name}.jsonl")
    summary = organizers.summarize_organizer(records)
    block = {**summary, "skipped": len(items) - summary["n"],
             "rotated": organizers.summarize_organizer(rotated) if rotated is not None else None}
    rot = f", rotated acc {100 * block['rotated']['accuracy']:.1f}%" if block["rotated"] else ""
    print(
        f"[{run_name}] {set_name} organizer acc {100 * summary['accuracy']:.1f}% (n={summary['n']}, "
        f"ABC mass {100 * summary['mean_abc_mass']:.1f}%{rot}) [{time.time() - started:.0f}s]"
    )
    return block


def _eval_general(model, tokenizer, cfg: dict, run_name: str) -> dict | None:
    ev = cfg["eval"]
    if not ev.get("general_loss"):
        return None
    heldout = get_dotted(cfg, "data.general_heldout")
    if not heldout or not resolve_path(heldout).exists():
        warnings.warn(f"general NLL skipped: data.general_heldout {heldout} not found", stacklevel=3)
        return None
    caps = [n for n in (ev.get("general_loss_max_items"), ev.get("limit")) if n]
    pairs = read_jsonl(resolve_path(heldout))
    pairs = pairs[: min(caps)] if caps else pairs
    result = general_loss.general_nll(
        model, tokenizer, pairs, batch_size=int(ev.get("batch_size") or 8), max_length=int(get_dotted(cfg, "model.max_length", 1024))
    )
    if not result["tokens"] or not math.isfinite(result["nll"]):
        warnings.warn(f"general NLL skipped: no scorable pairs in {heldout}", stacklevel=3)
        return None
    print(f"[{run_name}] general NLL {result['nll']:.4f} ({result['n']} pairs, {result['tokens']} tokens, {result['skipped']} skipped)")
    return result


def run_eval(
    model_path: str | Path,
    run_name: str,
    cfg: dict,
    *,
    is_base: bool = False,
    limit: int | None = None,
    model=None,
    tokenizer=None,
    force: bool = False,
) -> dict:
    """Evaluate model_path (or an already loaded model+tokenizer) and write results/<slug>/runs/<run_name>/.

    Returns the cached summary when fingerprint, model identity and weights signature match (unless force).
    A model loaded here is freed before returning. Base run names (base, base-limit<N>, base-dbg<fp8>) are reserved
    for is_base=True. Concurrent evals of the same run dir are serialized by a file lock (the second one then
    reuses the first one's summary instead of clearing its artifacts).
    """
    if not is_base and is_base_run_name(run_name):
        raise ValueError(f"run name {run_name!r} is reserved for the base model's eval (is_base=True)")
    started = time.time()
    if limit is not None:
        cfg = deep_merge(cfg, {"eval": {"limit": limit}})
    sets = eval_sets_from_cfg(cfg)
    if not sets:
        raise ValueError("nothing to evaluate: eval.sets is empty (or every optional set is missing)")
    out_dir = run_dir(cfg, run_name)
    identity, signature = model_identity(model_path), model_signature(model_path)
    load_info = model_load_info(model) if model is not None else planned_load_info(cfg)
    fingerprint, parts = _fingerprint(cfg, sets, load_info)

    with _eval_lock(out_dir):
        cached = load_summary(out_dir)
        if (
            cached is not None
            and not force
            and cached.get("fingerprint") == fingerprint
            and cached.get("model") == identity
            and cached.get("model_signature") == signature
        ):
            print(f"[{run_name}] reusing cached eval {out_dir / SUMMARY_FILE} (fingerprint {fingerprint})")
            return cached

        primary_set, primary_variant = _primary(cfg, sets)
        owned = model is None
        source = resolve_model_source(model_path) if owned or tokenizer is None else str(model_path)
        if owned:
            model, tokenizer = modeling.load_model_and_tokenizer(
                source,
                dtype=get_dotted(cfg, "model.dtype", "auto"),
                quantization=eval_quantization(cfg),
                attn_implementation=get_dotted(cfg, "model.attn_implementation", "auto"),
            )
            if model_load_info(model) != load_info:
                load_info = model_load_info(model)
                fingerprint, parts = _fingerprint(cfg, sets, load_info)
        elif tokenizer is None:
            tokenizer = modeling.load_tokenizer(source)

        try:
            _clear_run_artifacts(out_dir)
            set_results = _eval_sets(model, tokenizer, cfg, sets, out_dir, run_name)
            general = _eval_general(model, tokenizer, cfg, run_name)
        finally:
            if owned:
                del model  # drop our reference first, otherwise free_model cannot release the weights
                modeling.free_model(None)

        head = set_results[primary_set]["variants"][primary_variant] if primary_set else None
        summary = {
            "run_name": run_name,
            "model": identity,
            "model_source": model_identity(source),
            "model_signature": signature,
            "is_base": is_base,
            "base_model": cfg["model"]["base"],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": fingerprint,
            "fingerprint_parts": parts,
            **load_info,
            "primary_set": primary_set,
            "primary_variant": primary_variant,
            "headline": {k: head[k] for k in HEADLINE_KEYS} if head else None,
            "sets": set_results,
            "general_heldout": general,
            "elapsed_s": round(time.time() - started, 1),
        }
        _write_json_atomic(out_dir / SUMMARY_FILE, summary)
    return summary


# ----------------------------------------------------------------------------- CLI helpers (scripts/run_eval.py, scripts/compare.py)


def checkpoint_eval_overrides(model_path: str | Path) -> list[str]:
    """--set style overrides reproducing the eval settings (EVAL_CONFIG_KEYS) of the config a checkpoint was trained and
    evaluated with (<checkpoint>/resolved_config.yaml; [] when absent). A later eval without --config then lands on
    the pipeline's eval fingerprint (e.g. configs/bielik11b.yaml's eval.batch_size) instead of base.yaml's."""
    path = Path(model_path).expanduser()
    checkpoint = path.parent if path.name == "adapter" else path
    resolved = checkpoint / "resolved_config.yaml"
    if not resolved.is_file():
        return []
    with open(resolved, encoding="utf-8") as fh:
        saved = yaml.safe_load(fh) or {}

    def leaves(key: str, value) -> list[tuple[str, object]]:  # dotted leaves: a partial section must not replace base.yaml's
        return [kv for k, v in value.items() for kv in leaves(f"{key}.{k}", v)] if isinstance(value, dict) else [(key, value)]

    missing = object()
    pairs = [kv for key in EVAL_CONFIG_KEYS if (value := get_dotted(saved, key, missing)) is not missing for kv in leaves(key, value)]
    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in pairs]


def debug_suffix(cfg: dict, plain_cfg: dict, *, limit: int | None) -> str:
    """Run-name suffix that keeps CLI-narrowed debug evals from overwriting full ones: "-limit<N>" for a CLI --limit,
    "-dbg<fingerprint[:8]>" when CLI overrides (--set / --variants / --sets) changed the eval fingerprint of plain_cfg
    (the same config without them). "" for a plain eval."""
    suffix = f"-limit{limit}" if limit is not None else ""
    fingerprint = eval_fingerprint(cfg, planned_load_info(cfg))[0]
    try:
        plain = eval_fingerprint(plain_cfg, planned_load_info(plain_cfg))[0]
    except (FileNotFoundError, ValueError, RuntimeError):
        plain = None
    return suffix + ("" if fingerprint == plain else f"-dbg{fingerprint[:8]}")


# ----------------------------------------------------------------------------- reporting


def format_summary(summary: dict) -> str:
    """Human table: per set/variant points, score%, parse-fail%, lenient; loglik; general NLL; primary per-subject."""
    s = summary
    primary = (s["primary_set"], s["primary_variant"])
    lines = [
        f"{s['run_name']}  {s['model']}  [{s['device']}, {s['dtype']}, quant={s['quantization']}]  fingerprint {s['fingerprint']}",
        f"{'set/variant':<36}{'points':>15}{'score%':>9}{'parse-fail%':>13}{'lenient':>9}",
    ]
    for name, st in s["sets"].items():
        for variant, v in st["variants"].items():
            label = f"{name}/{variant}" + (" *" if (name, variant) == primary else "")
            points = f"{v['points']:g}/{v['max_points']:g}"
            lines.append(
                f"{label:<36}{points:>15}{v['score_pct']:>9.2f}{100 * v['parse_fail_rate']:>13.1f}{v['points_lenient']:>9g}"
            )
    for name, st in s["sets"].items():
        if st.get("loglik"):
            lines.append(f"loglik acc {name}: {100 * st['loglik']['acc']:.1f}% (n={st['loglik']['n']})")
    for name, st in s["sets"].items():
        if st.get("organizer"):
            org = st["organizer"]
            rot = f", rotated {100 * org['rotated']['accuracy']:.1f}%" if org.get("rotated") else ""
            lines.append(
                f"organizer acc {name}: {100 * org['accuracy']:.1f}% (n={org['n']}, skipped {org['skipped']}, "
                f"ABC mass {100 * org['mean_abc_mass']:.1f}%{rot})"
            )
    gen = s.get("general_heldout")
    lines.append(
        f"general NLL: {gen['nll']:.4f} ({gen['n']} pairs, {gen['tokens']} tokens)" if gen else "general NLL: not evaluated"
    )
    if primary[0] is None:
        lines.append("no prompt variants were graded (organizer-only eval): no headline, no per-subject points")
        return "\n".join(lines)
    per_subject = s["sets"][primary[0]]["variants"][primary[1]]["per_subject"]
    lines.append(f"per subject ({primary[0]}/{primary[1]}):")
    for subject, v in per_subject.items():
        points = f"{v['points']:g}/{v['max_points']:g}"
        lines.append(f"  {subject:<16}{points:>11}{v['score_pct']:>9.2f}%  lenient {v['points_lenient']:g}")
    return "\n".join(lines)
