#!/usr/bin/env python
"""History-matura LoRA SFT on prebuilt rows (hackathon final). Mixes in general replay, then trains the LoRA. It runs
no old 8-subject eval, merge or gate: the exam harness and the LLM judge score the model (docs/RUNBOOK.md).

  python scripts/train_history.py                                   # configs/history.yaml, default data paths
  python scripts/train_history.py --prepare-only                    # write the mixed files + stats.json, load no model
  python scripts/train_history.py --set run_name=history-v1 --set train.learning_rate=2e-4
  python scripts/train_history.py --train path/train.jsonl --dev path/dev.jsonl --protect-exam exams/<pack>/exam.json

Inputs
  --train  data/processed/history/train.jsonl   rows {"prompt": [msgs], "completion": [assistant]} or {"messages": [..., assistant]}
  --dev    data/processed/history/dev.jsonl     the matura eval split: eval_matura_loss after every epoch (never trained)
  data.general_pool (config)     general Polish replay, data.general_ratio of the train ROWS, answers <= general_max_completion_chars
  data.general_heldout (config)  PLLuM-Align rows: eval_general_heldout_loss after every epoch (forgetting check)

Outputs
  <paths.processed_root>/<run_name>/{train, val_matura, val_general, general_heldout}.jsonl + stats.json
  <paths.checkpoints_root>/<run_name>/{adapter/, train_metrics.json, resolved_config.yaml, training_meta.json}  (train.sft)

Next step: bash scripts/export_lora_gguf.sh checkpoints/<run_name>/adapter
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import re
import traceback
import unicodedata
from collections import Counter
from pathlib import Path

from eval.schema import read_jsonl, write_jsonl
from train.config import load_config, processed_dir, resolve_path
from train.dataset import file_sha256_16, general_count, general_row, user_prompt_key

DEFAULT_CONFIG = "configs/history.yaml"
DEFAULT_TRAIN = "data/processed/history/train.jsonl"
DEFAULT_DEV = "data/processed/history/dev.jsonl"
MIN_PROTECTED_CHARS = 40  # shorter exam questions ("Podaj nazwę...") would match unrelated rows


class DataError(ValueError):
    """Malformed or leaking training data."""


# ----------------------------------------------------------------------------- row normalisation


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def normalize_row(row: dict, where: str) -> dict:
    """Prebuilt row -> {"prompt", "completion", "kind", "id", "variant", "source_kind"}.

    kind is "general" for replay rows and "matura" for everything else, so train/sft.py's loss-token split
    (train_metrics.loss_tokens.matura_share) counts history rows as the exam signal."""
    if "prompt" in row and "completion" in row:
        prompt, completion = row["prompt"], row["completion"]
    elif row.get("messages"):
        prompt, completion = row["messages"][:-1], row["messages"][-1:]
    else:
        raise DataError(f"{where}: expected prompt/completion or messages, got keys {sorted(row)}")
    if not isinstance(prompt, list) or not prompt or not all(isinstance(m, dict) for m in prompt):
        raise DataError(f"{where}: prompt must be a non-empty list of chat messages")
    if not isinstance(completion, list) or len(completion) != 1 or completion[0].get("role") != "assistant":
        raise DataError(f"{where}: completion must be exactly one assistant message")
    if not (completion[0].get("content") or "").strip():
        raise DataError(f"{where}: empty assistant answer")
    if not any(m.get("role") == "user" for m in prompt):
        raise DataError(f"{where}: prompt has no user message")
    source_kind = row.get("kind")
    return {
        "prompt": prompt,
        "completion": completion,
        "kind": "general" if source_kind == "general" else "matura",
        "id": str(row.get("id") or where),
        "variant": row.get("variant"),
        "source_kind": source_kind,
    }


def load_rows(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{label} rows not found: {path} (build the history data first, see docs/RUNBOOK.md)")
    rows = [normalize_row(r, f"{path.name}:{i + 1}") for i, r in enumerate(read_jsonl(path))]
    if not rows:
        raise DataError(f"{path}: no rows")
    return rows


def user_text(row: dict) -> str:
    users = [m.get("content") or "" for m in row["prompt"] if m.get("role") == "user"]
    content = users[-1]
    if isinstance(content, list):  # multimodal-style content parts
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


# ----------------------------------------------------------------------------- leakage guards


def check_train_dev_disjoint(train: list[dict], dev: list[dict]) -> None:
    """Dev is the eval split: no dev row may appear in train (same id or identical user prompt)."""
    dev_ids = {r["id"] for r in dev}
    shared_ids = sorted({r["id"] for r in train} & dev_ids)
    dev_prompts = {_norm(user_text(r)) for r in dev}
    shared_prompts = [r["id"] for r in train if _norm(user_text(r)) in dev_prompts]
    if shared_ids or shared_prompts:
        raise DataError(f"train/dev overlap: {len(shared_ids)} shared ids {shared_ids[:5]}, "
                        f"{len(shared_prompts)} train rows with a dev prompt {shared_prompts[:5]}")


def protected_questions(exam_paths: list[str]) -> list[tuple[str, str]]:
    """(exam_id:item_id, normalized question) for every item of the given exam.json files."""
    out = []
    for p in exam_paths:
        exam = json.loads(Path(p).read_text(encoding="utf-8"))
        for item in exam.get("items", []):
            q = _norm(item.get("question") or "")
            if len(q) >= MIN_PROTECTED_CHARS:
                out.append((f"{exam.get('exam_id', Path(p).stem)}:{item.get('id')}", q))
    return out


def check_protected_exams(train: list[dict], exam_paths: list[str]) -> int:
    """Refuse train rows whose user prompt contains a question of a protected exam (e.g. the mock) verbatim."""
    questions = protected_questions(exam_paths)
    if not questions:
        return 0
    hits = []
    for r in train:
        text = _norm(user_text(r))
        hits += [(r["id"], key) for key, q in questions if q in text]
    if hits:
        raise DataError(f"{len(hits)} train rows contain a protected exam question verbatim, e.g. {hits[:5]}; "
                        "remove them from the training data (or drop --protect-exam if this is intended)")
    return len(questions)


# ----------------------------------------------------------------------------- mixing


def mix(cfg: dict, train: list[dict], dev: list[dict], out_dir: Path) -> dict:
    """Write train (history + replay, shuffled), val_matura (dev), val_general, general_heldout. Returns the data dict
    train_model() expects: {"train", "val_matura", "val_general", "general_heldout": Path, "stats": dict}."""
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 42))
    ratio = float(data_cfg.get("general_ratio") or 0.0)
    val_n = int(data_cfg.get("general_val_n") or 0)
    heldout_n = int(data_cfg.get("general_heldout_n") or 0)
    max_chars = data_cfg.get("general_max_completion_chars")

    heldout_path = resolve_path(data_cfg["general_heldout"]) if data_cfg.get("general_heldout") else None
    heldout_all = read_jsonl(heldout_path) if heldout_path and heldout_path.exists() else []
    n_general = general_count(len(train), ratio)
    pool_path = resolve_path(data_cfg["general_pool"]) if data_cfg.get("general_pool") else None
    if (n_general or val_n) and (pool_path is None or not pool_path.exists()):
        raise FileNotFoundError(f"data.general_pool not found: {data_cfg.get('general_pool')} "
                                "(python scripts/fetch_external.py --only general, or --set data.general_ratio=0)")
    pool = read_jsonl(pool_path) if pool_path and pool_path.exists() else []
    heldout_prompts = {user_prompt_key(p) for p in heldout_all}
    usable = [p for p in pool if user_prompt_key(p) not in heldout_prompts
              and (max_chars is None or len(p["messages"][-1].get("content") or "") <= int(max_chars))]
    order = list(range(len(usable)))
    random.Random(f"{seed}:general").shuffle(order)
    val_general = [usable[i] for i in order[:val_n]]
    general_train = [usable[i] for i in order[val_n:val_n + n_general]]
    if len(general_train) < n_general:
        print(f"[train_history] WARNING: wanted {n_general} replay rows, only {len(general_train)} available")

    rows = train + [general_row(p) for p in general_train]
    random.Random(f"{seed}:shuffle").shuffle(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"{name}.jsonl" for name in ("train", "val_matura", "val_general", "general_heldout")}
    write_jsonl(rows, paths["train"])
    write_jsonl(dev, paths["val_matura"])
    write_jsonl([general_row(p) for p in val_general], paths["val_general"])
    write_jsonl([general_row(p) for p in heldout_all[:heldout_n]], paths["general_heldout"])

    answer_chars = sorted(len(r["completion"][0]["content"]) for r in train)
    stats = {
        "run_name": cfg["run_name"],
        "seed": seed,
        "n_history_train": len(train),
        "n_history_dev": len(dev),
        "n_general_train": len(general_train),
        "general_rows_wanted": n_general,
        "general_ratio": ratio,
        "general_ratio_built": round(len(general_train) / len(rows), 4) if rows else 0.0,
        "n_train_rows": len(rows),
        "n_val_general": len(val_general),
        "n_general_heldout": min(heldout_n, len(heldout_all)),
        "general_pool_usable": len(usable),
        "source_kinds_train": dict(sorted(Counter(str(r["source_kind"]) for r in train).items())),
        "source_kinds_dev": dict(sorted(Counter(str(r["source_kind"]) for r in dev).items())),
        "history_answer_chars": {"min": answer_chars[0], "median": answer_chars[len(answer_chars) // 2],
                                 "max": answer_chars[-1]} if answer_chars else None,
        "files": {name: str(p) for name, p in paths.items()},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**paths, "stats": stats}


# ----------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"run config (default {DEFAULT_CONFIG}; inherits configs/base.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override (repeatable)")
    ap.add_argument("--train", default=DEFAULT_TRAIN, help=f"history train rows (default {DEFAULT_TRAIN})")
    ap.add_argument("--dev", default=DEFAULT_DEV, help=f"history dev rows = matura eval split (default {DEFAULT_DEV})")
    ap.add_argument("--protect-exam", action="append", default=[], metavar="EXAM_JSON",
                    help="exam.json whose questions must not appear in train rows (repeatable; e.g. the mock pack)")
    ap.add_argument("--prepare-only", action="store_true", help="write the mixed files + stats.json and exit (no model)")
    args = ap.parse_args(argv)
    bad = [s for s in args.set if "=" not in s]
    if bad:
        ap.error(f"--set expects KEY=VALUE, got {bad}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    train_path, dev_path = resolve_path(args.train), resolve_path(args.dev)
    out_dir = processed_dir(cfg)
    if out_dir.resolve() in (train_path.parent.resolve(), dev_path.parent.resolve()):
        raise DataError(f"run_name {cfg['run_name']!r} writes to {out_dir}, the folder of the input rows; "
                        "pick another run_name (--set run_name=...)")

    train, dev = load_rows(train_path, "train"), load_rows(dev_path, "dev")
    check_train_dev_disjoint(train, dev)
    n_protected = check_protected_exams(train, args.protect_exam)
    data = mix(cfg, train, dev, out_dir)
    data["stats"]["inputs"] = {
        "train": {"path": str(args.train), "sha256_16": file_sha256_16(train_path)},
        "dev": {"path": str(args.dev), "sha256_16": file_sha256_16(dev_path)},
        "protected_exams": list(args.protect_exam),
        "protected_questions_checked": n_protected,
    }
    (out_dir / "stats.json").write_text(json.dumps(data["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data["stats"], ensure_ascii=False, indent=2))
    if args.prepare_only:
        print(f"[train_history] prepared {out_dir} (no training: --prepare-only)")
        return 0

    from train.sft import train_model  # heavy imports (torch, transformers) only when training

    result = train_model(cfg, data)
    metrics = result["train_metrics"]
    last = metrics["epochs"][-1] if metrics["epochs"] else {}
    print(f"\n[train_history] run {cfg['run_name']}: adapter {result['adapter_dir']}")
    print(f"  step0 (base) losses: {metrics['step0']}")
    print(f"  last epoch losses:   { {k: last.get(k) for k in ('train_loss', 'matura_loss', 'general_loss', 'general_heldout_loss')} }")
    print(f"  dropped_too_long_by_kind (train): {metrics['dropped_too_long_by_kind']}  "
          f"matura share of loss tokens: {metrics['loss_tokens']['matura_share']}  runtime {metrics['runtime_s']} s")
    if metrics["dropped_too_long"]:
        print(f"  WARNING: {metrics['dropped_too_long']} train rows exceeded model.max_length={cfg['model']['max_length']} "
              "and were not trained (essays?): raise model.max_length or shorten those rows")
    print(f"next: bash scripts/export_lora_gguf.sh {result['adapter_dir']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("train_history.py: interrupted")
    except (FileNotFoundError, DataError) as exc:  # bad inputs: the message says what to fix
        sys.exit(f"train_history.py: FAILED: {exc}")
    except Exception as exc:
        traceback.print_exc()
        sys.exit(f"train_history.py: FAILED: {type(exc).__name__}: {exc}")
