"""YAML config loading with `inherit:` chains, deep merge, and dotted CLI overrides."""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import re

import yaml

_SCI = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _coerce_sci(obj: Any) -> Any:
    """YAML 1.1 loads '1e-4' as a string; convert such scalars (anywhere in the tree) to float."""
    if isinstance(obj, dict):
        return {k: _coerce_sci(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_sci(v) for v in obj]
    if isinstance(obj, str) and _SCI.match(obj.strip()):
        return float(obj)
    return obj

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DEFAULT_CONFIG = CONFIGS / "base.yaml"

# Weight tables: a later config replaces the whole table (a variant it leaves out gets weight 0), exactly like
# `--set data.variant_mix={...}` does, instead of merging into base.yaml's table.
ATOMIC_KEYS = frozenset({"variant_mix"})
_FREE_TOP = frozenset({"run_name", "inherit"})  # top-level keys a config may set although base.yaml does not
_FREE_DICTS = ("data.variant_mix",)  # dict-valued settings whose keys are data, not config keys
_MISSING = object()


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; values under ATOMIC_KEYS (and all non-dicts) are replaced, not merged."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ATOMIC_KEYS:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_chain(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config inherit cycle at {path}")
    seen.add(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        parent_path = (path.parent / parent) if not Path(parent).is_absolute() else Path(parent)
        return deep_merge(_load_chain(parent_path, seen), cfg)
    return cfg


def parse_value(raw: str) -> Any:
    """Parse a CLI override value with YAML semantics (numbers, bools, null, lists, dicts)."""
    try:
        val = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(val, str):  # YAML 1.1 reads "1e-4" as a string; accept scientific notation as float
        try:
            return float(val) if any(c in val for c in ".eE") else int(val)
        except ValueError:
            return val
    return val


def parse_overrides(overrides: list[str] | dict | None) -> list[tuple[str, Any]]:
    """["train.learning_rate=1e-4"] or {"train.learning_rate": 1e-4} -> [(dotted key, parsed value)] in order."""
    if not overrides:
        return []
    if isinstance(overrides, dict):
        return [(str(k).strip(), parse_value(v) if isinstance(v, str) else v) for k, v in overrides.items()]
    pairs = []
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be KEY=VALUE")
        key, raw = item.split("=", 1)
        pairs.append((key.strip(), parse_value(raw)))
    return pairs


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _leaf_keys(d: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        out += _leaf_keys(v, key + ".") if isinstance(v, dict) and v and key not in _FREE_DICTS else [key]
    return out


def check_keys(defaults: dict, keys: Iterable[str], where: str) -> None:
    """Raise ValueError on dotted keys that configs/base.yaml does not define (typo guard), with a did-you-mean hint.
    Without it `--set train.learnig_rate=2e-4` would silently train with the default learning rate."""
    known = _leaf_keys(defaults)
    for key in keys:
        if key.split(".")[0] in _FREE_TOP or key.startswith("_") or any(key.startswith(p + ".") for p in _FREE_DICTS):
            continue
        if get_dotted(defaults, key, _MISSING) is _MISSING:
            hint = difflib.get_close_matches(key, known, n=1)
            raise ValueError(f"{where}: unknown config key {key!r}" + (f" (did you mean {hint[0]!r}?)" if hint else "")
                             + " (every key must exist in configs/base.yaml)")


def load_config(path: str | Path | None = None, overrides: list[str] | dict | None = None) -> dict:
    """Load a config file (default configs/base.yaml), resolving `inherit`, then apply overrides.

    overrides: ["train.learning_rate=1e-4", "lora.r=32"] or {"train.learning_rate": 1e-4}.
    Keys of a config file and of overrides must exist in configs/base.yaml (ValueError otherwise).
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = _load_chain(path, set())
    defaults = _load_chain(DEFAULT_CONFIG, set()) if DEFAULT_CONFIG.exists() else None
    if defaults is not None and path.resolve() != DEFAULT_CONFIG.resolve():
        # every config implicitly inherits base.yaml defaults for keys it does not set
        check_keys(defaults, _leaf_keys(cfg), str(path))
        cfg = deep_merge(defaults, cfg)
    pairs = parse_overrides(overrides)
    if defaults is not None:
        check_keys(defaults, [k for k, _ in pairs], "override")
    for key, value in pairs:
        set_dotted(cfg, key, value)
    cfg = _coerce_sci(cfg)
    cfg.setdefault("run_name", path.stem)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def dump_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, fh, allow_unicode=True, sort_keys=False)


# ----------------------------------------------------------------------------- shared paths


def base_slug(base_model: str) -> str:
    """'speakleash/Bielik-4.5B-v3.0-Instruct' -> 'bielik-4.5b-v3.0-instruct' (results are namespaced per base model)."""
    return str(base_model).rstrip("/").split("/")[-1].lower()


def results_dir(cfg: dict) -> Path:
    return resolve_path(get_dotted(cfg, "paths.results_root", "results")) / base_slug(cfg["model"]["base"])


def run_dir(cfg: dict, run_name: str | None = None) -> Path:
    return results_dir(cfg) / "runs" / (run_name or cfg["run_name"])


def checkpoints_root(cfg: dict) -> Path:
    return resolve_path(get_dotted(cfg, "paths.checkpoints_root", "checkpoints"))


def checkpoint_dir(cfg: dict, run_name: str | None = None) -> Path:
    return checkpoints_root(cfg) / (run_name or cfg["run_name"])


def processed_dir(cfg: dict, run_name: str | None = None) -> Path:
    return resolve_path(get_dotted(cfg, "paths.processed_root", "data/processed")) / (run_name or cfg["run_name"])


def candidate_link_name(slug: str) -> str:
    """Stable symlink to one base model's candidate checkpoint: checkpoint dirs are shared by every base model, so
    each base slug gets its own link (<checkpoints_root>/CANDIDATE-<base_slug>)."""
    return f"CANDIDATE-{slug}"


# ----------------------------------------------------------------------------- run names

BASE_RUN = "base"
_BASE_RUN_NAME = re.compile(r"base(?:-limit\d+)?(?:-dbg[0-9a-f]{8})?")
_DEBUG_SUFFIX = re.compile(r"-(?:limit\d+(?:-dbg[0-9a-f]{8})?|dbg[0-9a-f]{8})$")


def is_base_run_name(name: str) -> bool:
    """'base' and its debug variants (base-limit<N>, base-dbg<fp8>, base-limit<N>-dbg<fp8>) hold base-model evals only."""
    return bool(_BASE_RUN_NAME.fullmatch(name))


def is_debug_run_name(name: str) -> bool:
    """Names of CLI-narrowed debug evals (<name>-limit<N> and/or -dbg<fp8>): never recorded in the registry."""
    return bool(_DEBUG_SUFFIX.search(name))


_LOCAL_ROOT_NAMES = {"checkpoints", ".smoke", "results", "data", "export", "exports", "outputs"}


def looks_like_local_path(model: str) -> bool:
    """True when `model` is meant as a local path even if it does not exist yet (so a missing checkpoint gives a clear
    'not found' instead of a confusing Hugging Face hub lookup). Hub ids are 'name' or 'org/name'; a gitignored
    checkpoints/ dir is absent in a fresh clone, so its name alone must be enough."""
    if model.startswith((".", "/", "~")):
        return True
    parts = Path(model).parts
    if len(parts) > 2:
        return True
    if len(parts) == 2:
        first = parts[0]
        return first in _LOCAL_ROOT_NAMES or Path(first).is_dir() or (ROOT / first).is_dir()
    return False
