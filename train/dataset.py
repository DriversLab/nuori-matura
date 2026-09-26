"""Processed training data for one run: synthetic matura items + general Polish instruction pairs -> TRL rows.

Deterministic from cfg.seed. Every stage draws from its own seeded RNG stream, so e.g. changing general_ratio
does not change the matura train/val split or its renders. Matura rows are built with eval.prompts
(the same prompt/answer-format code the evaluator uses). Outputs go to train.config.processed_dir(cfg):

    train.jsonl         shuffled matura renders + general rows   {"prompt", "completion", "kind", "id", "variant"}
    val_matura.jsonl    held-back synthetic items (whole source articles), canonical variant, no system prompt, original option order
    val_general.jsonl   general_val_n pool rows disjoint from the train general rows
    general_heldout.jsonl  first general_heldout_n rows of data.general_heldout (PLLuM-Align)
    stats.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from eval.answer_format import VARIANTS
from eval.prompts import build_training_example, render_task
from eval.schema import LETTERS, load_items, read_jsonl, validate_item, write_jsonl
from train.config import ROOT, config_hash, processed_dir, resolve_path

log = logging.getLogger(__name__)

# Items in these files must never be trained on (checked by id and by identical rendered task text).
PROTECTED_EVAL_FILES = ("data/eval/heldout.jsonl", "data/eval/dev.jsonl", "data/eval/ext_llmzszl_matura.jsonl",
                        "data/eval/ext_prawko_dev.jsonl", "data/eval/ext_prawko_test.jsonl")
BLOCKLIST_DIR = "data/blocklist"
DEDUP_REPORT = "dedup_report.json"

# A standalone option letter (not part of a word: "witamina C" and "25°C" count, "Celina" does not).
_L = r"(?<![^\W\d_])[A-H](?![^\W\d_])"
# Texts that refer to options by letter: re-lettering would change their meaning. Checked in options, question,
# context and the left column of match items ("tylko A", "zarówno A, jak i B", "A–C", "Wybierz odpowiedź A, jeśli ...").
_LETTER_REFERENCE = re.compile(
    rf"{_L}\s*(?:i|oraz|,|lub|albo|[–—-])\s*{_L}"
    rf"|(?i:odpowied\w*|opcj\w*|wariant\w*|liter\w*|tylko|wyłącznie|jedynie|zarówno|ani|oprócz|poza|wyjątkiem|jak\s+i)\s+{_L}"
)
# Option texts that refer to other options by position ("żadne z powyższych"); in a question "poniższych" means the
# option set as a whole, which re-lettering does not change.
_POSITION_REFERENCE = re.compile(r"(?i:powyższ|poniższ)")
_STANDALONE_LETTER = re.compile(_L)


class LeakageError(ValueError):
    """A protected evaluation item appeared among the training/validation items (or the dedup is stale)."""


# ----------------------------------------------------------------------------- helpers


def _rng(seed: int, stage: str) -> random.Random:
    return random.Random(f"{seed}:{stage}")


def file_sha256_16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _norm_task(item: dict) -> str:
    return re.sub(r"\s+", " ", render_task(item)).strip().casefold()


def user_prompt_key(pair: dict) -> str:
    """Normalized content of the last user message of a general pair (for heldout/pool overlap checks; NFKC like
    train.external.normalize_prompt at fetch time)."""
    user = [m.get("content") or "" for m in pair["messages"] if m.get("role") == "user"]
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", user[-1] if user else "")).strip().casefold()


def _portable(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def eval_set_paths(cfg: dict) -> list[str]:
    """Existing files of cfg eval.sets (dict or plain-string entries)."""
    entries = (cfg.get("eval") or {}).get("sets") or []
    paths = [str(e["path"]) if isinstance(e, dict) else str(e) for e in entries]
    return [p for p in paths if resolve_path(p).exists()]


def check_no_eval_leak(items: list[dict], protected_files: tuple[str, ...] = PROTECTED_EVAL_FILES) -> None:
    """Raise LeakageError if any item id (or identical rendered task) belongs to a protected eval file."""
    ids: dict[str, str] = {}
    tasks: dict[str, str] = {}
    for rel in protected_files:
        path = resolve_path(rel)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            ids[row["id"]] = rel
            if row.get("type") and row.get("question"):
                tasks[_norm_task(row)] = f"{rel}:{row['id']}"
    by_id = [f"{it['id']} ({ids[it['id']]})" for it in items if it["id"] in ids]
    by_text = [f"{it['id']} == {tasks[_norm_task(it)]}" for it in items if _norm_task(it) in tasks]
    if by_id or by_text:
        raise LeakageError(f"protected eval items among training data: ids={by_id[:10]} identical_tasks={by_text[:10]}")


def dedup_staleness(synthetic_path: Path, protected_paths: list[Path]) -> dict | None:
    """Compare the protected-file hashes that scripts/build_data.py recorded in dedup_report.json (next to the synthetic
    file) with the current files. None when the report or its hashes are missing (nothing to compare); else
    {"changed": [recorded files that changed or disappeared], "not_deduplicated": [current protected files never
    deduplicated against]}."""
    report_path = synthetic_path.parent / DEDUP_REPORT
    if not report_path.exists():
        return None
    recorded = (json.loads(report_path.read_text(encoding="utf-8")).get("inputs") or {}).get("protected_files")
    if not recorded:
        return None
    changed = [rel for rel, digest in recorded.items() if not resolve_path(rel).exists() or file_sha256_16(resolve_path(rel)) != digest]
    known = {resolve_path(rel).resolve() for rel in recorded}
    missing = [_portable(p) for p in dict.fromkeys(protected_paths) if p.exists() and p.resolve() not in known]
    return {"changed": changed, "not_deduplicated": missing}


def split_by_subject(items: list[dict], val_fraction: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Group-aware stratified split: items sharing `source` (the same Wikipedia article: sibling items built from the
    same facts) never straddle train/val. n_val = round(N * f) (>= 1 when f > 0 and N >= 2) is allocated to subjects by
    largest remainder and met with whole groups (it may overshoot by less than one group); every subject keeps at least
    one train group. Items without a source are their own group. Returned lists keep the input order."""
    if val_fraction <= 0 or len(items) < 2:
        return list(items), []
    groups: dict[str, dict[str, list[int]]] = defaultdict(dict)  # subject -> group key -> item indices
    for i, it in enumerate(items):
        groups[it["subject"]].setdefault(it.get("source") or it["id"], []).append(i)
    sizes = {s: sum(map(len, g.values())) for s, g in groups.items()}
    n_val = max(1, round(len(items) * val_fraction))
    exact = {s: n * n_val / len(items) for s, n in sizes.items()}
    quota = {s: min(int(exact[s]), sizes[s] - 1) for s in groups}
    for s in sorted(groups, key=lambda s: (-(exact[s] - int(exact[s])), s)):
        if sum(quota.values()) >= n_val:
            break
        if quota[s] < sizes[s] - 1:
            quota[s] += 1
    val_idx: set[int] = set()
    for s in sorted(groups):
        keys = list(groups[s])
        rng.shuffle(keys)
        taken = 0
        for key in keys[: len(keys) - 1]:  # at least one group stays in train
            if taken >= quota[s]:
                break
            val_idx.update(groups[s][key])
            taken += len(groups[s][key])
    return [it for i, it in enumerate(items) if i not in val_idx], [it for i, it in enumerate(items) if i in val_idx]


def shuffle_options(item: dict, rng: random.Random) -> tuple[dict, bool]:
    """Randomly re-letter mc/multi options (or the right column of match) and remap the answer.

    Returns (item, shuffled). Items are left unchanged when an option refers to other options by letter or position,
    or when the question, context or match left column refers to options by letter.
    """
    t = item["type"]
    field = {"mc": "options", "multi": "options", "match": "right"}.get(t)
    if field is None:
        return item, False
    options = list(item[field].values())
    texts = [*options, *item.get("left", []), item["question"], item.get("context") or ""]
    if any(_POSITION_REFERENCE.search(v) for v in options) or any(_LETTER_REFERENCE.search(v) for v in texts):
        return item, False
    old_letters = list(item[field])
    order = old_letters[:]
    rng.shuffle(order)
    new = copy.deepcopy(item)
    new[field] = {LETTERS[i]: item[field][old] for i, old in enumerate(order)}
    remap = {old: LETTERS[i] for i, old in enumerate(order)}
    if t == "mc":
        new["answer"] = remap[item["answer"]]
    elif t == "multi":
        new["answer"] = sorted(remap[a] for a in item["answer"])
    else:
        new["answer"] = {k: remap[v] for k, v in item["answer"].items()}
    return validate_item(new), order != old_letters


def _variant_sampler(mix: dict[str, float]) -> tuple[list[str], list[float]]:
    unknown = set(mix) - set(VARIANTS)
    if unknown:
        raise ValueError(f"data.variant_mix has unknown variants {sorted(unknown)}; known: {sorted(VARIANTS)}")
    names = [v for v, w in mix.items() if w and w > 0]
    if not names:
        raise ValueError("data.variant_mix needs at least one variant with a positive weight")
    return names, [float(mix[v]) for v in names]


def matura_row(item: dict, variant: str, system_prompt: str | None) -> dict:
    return {**build_training_example(item, variant, system_prompt), "kind": "matura", "id": item["id"], "variant": variant}


def general_row(pair: dict) -> dict:
    msgs = pair["messages"]
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError(f"general pair {pair.get('id')}: last message must be the assistant response")
    return {"prompt": msgs[:-1], "completion": [msgs[-1]], "kind": "general", "id": pair["id"], "variant": None}


def general_count(n_matura_rows: int, ratio: float) -> int:
    """Number of general rows so that general / (general + matura) == ratio (rows, not loss tokens)."""
    if not 0 <= ratio < 1:
        raise ValueError(f"data.general_ratio must be in [0, 1), got {ratio}")
    return round(n_matura_rows * ratio / (1 - ratio))


def render_train(items: list[dict], data_cfg: dict, rng: random.Random) -> tuple[list[dict], Counter]:
    names, weights = _variant_sampler(data_cfg.get("variant_mix") or {"canonical": 1.0})
    prompts = list(data_cfg.get("system_prompts") or [])
    prob = float(data_cfg.get("system_prompt_prob") or 0.0)
    do_shuffle = bool(data_cfg.get("shuffle_options", False))
    renders = int(data_cfg.get("renders_per_item") or 1)
    counts: Counter = Counter()
    rows: list[dict] = []
    for item in items:
        for _ in range(renders):
            variant = rng.choices(names, weights)[0]
            if variant == "reason_then_answer" and not (item.get("rationale") or "").strip():
                variant = "canonical"
                counts["reason_fallback_canonical"] += 1
            system_prompt = rng.choice(prompts) if prompts and rng.random() < prob else None
            rendered = item
            if do_shuffle:
                rationale_cites_letters = variant == "reason_then_answer" and _STANDALONE_LETTER.search(item.get("rationale") or "")
                if rationale_cites_letters:
                    counts["shuffle_skipped_rationale_letters"] += 1
                else:
                    rendered, shuffled = shuffle_options(item, rng)
                    counts["options_shuffled"] += int(shuffled)
            counts[f"variant:{variant}"] += 1
            counts["with_system_prompt"] += int(system_prompt is not None)
            if rendered["type"] == "mc":
                counts[f"mc_answer:{rendered['answer']}"] += 1
            rows.append(matura_row(rendered, variant, system_prompt))
    return rows, counts


def _load_general(path_key: str, data_cfg: dict, needed: bool) -> list[dict]:
    rel = data_cfg.get(path_key)
    path = resolve_path(rel) if rel else None
    if path is None or not path.exists():
        if needed:
            raise FileNotFoundError(f"data.{path_key} not found: {rel} (run: python scripts/fetch_external.py --only general)")
        return []
    return read_jsonl(path)


# ----------------------------------------------------------------------------- main entry


def build_processed(cfg: dict, out_dir: str | Path | None = None, *, allow_stale_dedup: bool = False) -> dict:
    """Build train/val/general files for cfg. Returns {"train", "val_matura", "val_general", "general_heldout": Path, "stats": dict}.

    Raises LeakageError when a synthetic item is a protected eval item (heldout/dev/ext or any eval.sets file, by id or
    identical task), or when dedup_report.json shows clean.jsonl was deduplicated against different eval/blocklist files
    than the current ones (allow_stale_dedup=True logs a warning instead).
    """
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 42))
    out = Path(out_dir) if out_dir else processed_dir(cfg)

    synthetic_path = resolve_path(data_cfg["synthetic"])
    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"data.synthetic not found: {data_cfg['synthetic']}. Build the cleaned synthetic items first: "
            "python scripts/build_data.py"
        )
    all_items = load_items(synthetic_path)
    items = all_items
    n_matura = data_cfg.get("n_matura")
    if n_matura is not None and int(n_matura) < len(all_items):
        keep = set(_rng(seed, "n_matura").sample(range(len(all_items)), int(n_matura)))
        items = [it for i, it in enumerate(all_items) if i in keep]
    eval_paths = eval_set_paths(cfg)
    check_no_eval_leak(items, tuple(dict.fromkeys((*PROTECTED_EVAL_FILES, *eval_paths))))
    protected = [resolve_path(p) for p in (*PROTECTED_EVAL_FILES, *eval_paths)] + sorted(resolve_path(BLOCKLIST_DIR).glob("*.jsonl"))
    stale = dedup_staleness(synthetic_path, protected)
    if stale and (stale["changed"] or stale["not_deduplicated"]):
        msg = (f"{data_cfg['synthetic']} was deduplicated against different eval/blocklist files: changed={stale['changed']} "
               f"not_deduplicated={stale['not_deduplicated']}; rerun python scripts/build_data.py "
               "(--extra-protected <file> for a new eval set)")
        if not allow_stale_dedup:
            raise LeakageError(msg)
        log.warning(msg)

    train_items, val_items = split_by_subject(items, float(data_cfg.get("val_fraction") or 0.0), _rng(seed, "split"))
    train_rows, render_counts = render_train(train_items, data_cfg, _rng(seed, "render"))
    val_rows = [matura_row(it, "canonical", None) for it in val_items]

    ratio = float(data_cfg.get("general_ratio") or 0.0)
    val_n = int(data_cfg.get("general_val_n") or 0)
    heldout_n = int(data_cfg.get("general_heldout_n") or 0)
    max_chars = data_cfg.get("general_max_completion_chars")
    n_general = general_count(len(train_rows), ratio)
    heldout_all = _load_general("general_heldout", data_cfg, needed=heldout_n > 0)
    heldout = heldout_all[:heldout_n]
    pool = _load_general("general_pool", data_cfg, needed=n_general > 0 or val_n > 0)
    heldout_prompts = {user_prompt_key(p) for p in heldout_all}  # the gate scores up to eval.general_loss_max_items rows of the file
    not_heldout = [p for p in pool if user_prompt_key(p) not in heldout_prompts]
    usable = [p for p in not_heldout if max_chars is None or len(p["messages"][-1].get("content") or "") <= int(max_chars)]
    order = list(range(len(usable)))
    _rng(seed, "general").shuffle(order)
    val_general = [usable[i] for i in order[:val_n]]
    remaining = order[val_n:]
    if n_general > len(remaining):
        log.warning("general pool too small: wanted %d train rows, only %d available after val/heldout/length cap", n_general, len(remaining))
    general_train = [usable[i] for i in remaining[:n_general]]

    rows = train_rows + [general_row(p) for p in general_train]
    _rng(seed, "shuffle").shuffle(rows)

    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}.jsonl" for name in ("train", "val_matura", "val_general", "general_heldout")}
    write_jsonl(rows, paths["train"])
    write_jsonl(val_rows, paths["val_matura"])
    write_jsonl([general_row(p) for p in val_general], paths["val_general"])
    write_jsonl([general_row(p) for p in heldout], paths["general_heldout"])

    stats = {
        "run_name": cfg.get("run_name"),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "inputs": {
            "synthetic": {"path": str(data_cfg["synthetic"]), "sha256_16": file_sha256_16(synthetic_path), "n_items": len(all_items)},
            "general_pool": {"path": data_cfg.get("general_pool"), "n_rows": len(pool),
                             "excluded_heldout_prompts": len(pool) - len(not_heldout),
                             "general_max_completion_chars": max_chars, "excluded_long_completions": len(not_heldout) - len(usable)},
            "general_heldout": {"path": data_cfg.get("general_heldout"), "n_rows": len(heldout)},
        },
        "eval_sets_leak_checked": eval_paths,
        "dedup_fresh": None if stale is None else not (stale["changed"] or stale["not_deduplicated"]),
        "dedup_stale": stale if stale and (stale["changed"] or stale["not_deduplicated"]) else None,
        "n_matura_items": len(items),
        "n_train_items": len(train_items),
        "n_val_items": len(val_items),
        "n_val_sources": len({it.get("source") or it["id"] for it in val_items}),
        "per_subject": {s: {"train": sum(it["subject"] == s for it in train_items), "val": sum(it["subject"] == s for it in val_items)}
                        for s in sorted({it["subject"] for it in items})},
        "per_type_train": dict(sorted(Counter(it["type"] for it in train_items).items())),
        "renders_per_item": int(data_cfg.get("renders_per_item") or 1),
        "n_train_matura_rows": len(train_rows),
        "n_train_general_rows": len(general_train),
        "general_rows_wanted": n_general,
        "n_train_rows": len(rows),
        "general_ratio": ratio,
        # before sft's max_length filter (mostly drops long general rows): train_metrics.general_ratio_trained is what trained
        "general_ratio_built": round(len(general_train) / len(rows), 4) if rows else 0.0,
        "n_val_matura": len(val_rows),
        "n_val_general": len(val_general),
        "n_general_heldout": len(heldout),
        "render": dict(sorted(render_counts.items())),
        "processed_dir": str(out),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("processed data: %d train rows (%d matura, %d general), %d val_matura, %d val_general -> %s",
             len(rows), len(train_rows), len(general_train), len(val_rows), len(val_general), out)
    return {**paths, "stats": stats}
