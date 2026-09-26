#!/usr/bin/env python
"""Generate the judge-facing write-up (what we tried, what worked, why) from committed data + the results registry.

  python scripts/writeup.py                            # base model from configs/base.yaml -> results/<slug>/WRITEUP.md
  python scripts/writeup.py --config configs/bielik11b.yaml
  python scripts/writeup.py --stdout

Every number is read from files (verification log, dedup report, fetch report, runs.jsonl, summaries); nothing is
typed by hand, so re-running after each run keeps the story current. No model is loaded.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import datetime as dt
import json
from collections import Counter

from eval.answer_format import ANSWER_PREFIX, FORMAT_VERSION, VARIANTS
from eval.schema import read_jsonl
from train.config import load_config, resolve_path, results_dir


def _load_json(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _rows(path: pathlib.Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def _fmt(x, nd: int = 2, sign: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _verification_sentence(st: dict) -> str:
    skipped, unverified = st.get("verify_skipped", 0), st.get("unverified", 0)
    scope = "every item" if not (skipped or unverified) else f"{st['raw'] - skipped - unverified} items ({skipped} shards' items exempt, {unverified} unanswered and dropped)"
    return (f"- **Blind open-book verification**: an independent solver answered {scope} without seeing the key; dropped "
            f"{st.get('disagree', 0)} disagreements, {st.get('low_conf', 0)} low-confidence and {st.get('concern', 0)} items the "
            "agreeing solver still flagged (ambiguity, debatable statement, typo or low matura relevance).")


def _v(x) -> str:
    return "—" if x is None else str(x)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def section_eval() -> list[str]:
    heldout = _rows(ROOT / "data/eval/heldout.jsonl")
    dev = _rows(ROOT / "data/eval/dev.jsonl")
    ext = _rows(ROOT / "data/eval/ext_llmzszl_matura.jsonl")
    general = _rows(ROOT / "data/eval/general_heldout.jsonl")
    log = _rows(ROOT / "data/eval/verification_log.jsonl")
    decisions = Counter(r["decision"] for r in log)
    types = Counter(i["type"] for i in heldout)
    subjects = Counter(i["subject"] for i in heldout)
    points = sum(i.get("points", 1) for i in heldout)
    out = [
        "## 1. Evaluation harness (built first)",
        "",
        f"- **Held-out set** (`data/eval/heldout.jsonl`): {len(heldout)} original matura-style items, {points} points, "
        f"{len(subjects)} subjects ({', '.join(f'{s} {n}' for s, n in sorted(subjects.items()))}); formats "
        f"{', '.join(f'{t} {n}' for t, n in types.most_common())}. Plus {len(dev)} dev items for prompt iteration.",
        f"- **Answer-key verification**: {len(log)} authored items, each solved by two independent blind solvers "
        f"(closed-book and web-verified) and adjudicated by a critic: {decisions.get('keep', 0)} kept as-is, "
        f"{decisions.get('keep_fixed', 0)} fixed (mostly widened accepted short-answer variants), {decisions.get('drop', 0)} dropped.",
        f"- **External real-exam set**: {len(ext)} CKE matura multiple-choice items (amu-cai/llmzszl-dataset; math/physics/biology) "
        "as a secondary, much larger check against regression on real exam questions.",
        f"- **General capability check**: NLL on {len(general)} human-annotated Polish instruction pairs (NASK-PIB/PLLuM-Align), "
        "a source disjoint from the training mix.",
        f"- **Grading** (format {FORMAT_VERSION}): strict parsing of a single `{ANSWER_PREFIX} X` line is the headline "
        "(assumed official behaviour); a lenient parser recovers answers from free text so every delta splits into "
        "*format* and *knowledge* components. MC log-likelihood accuracy (lm-eval style) is reported alongside.",
        "- **Prompt-robustness variants**: " + "; ".join(f"`{k}` — {v.description}" for k, v in VARIANTS.items() if k != "reason_then_answer") + ".",
        "- **Comparison**: every run is compared with the untouched base under an identical eval fingerprint (items, grading code, "
        "generation settings, precision, library versions), with paired bootstrap CI and sign test.",
        "",
    ]
    return out


def section_data() -> list[str]:
    fetch = _load_json(ROOT / "data/wiki/fetch_report.json") or {}
    report = _load_json(ROOT / "data/synthetic/dedup_report.json")
    pool = _rows(ROOT / "data/general/train_pool.jsonl")
    out = ["## 2. Training data", ""]
    if fetch:
        per = fetch.get("per_subject", {})
        out.append(f"- **Source**: {fetch.get('total_articles', '?')} Polish Wikipedia articles curated against the matura syllabus "
                   f"({', '.join(f'{s} {n}' for s, n in per.items())}).")
    if report:
        st = report["stages"]
        protected = {k: v for k, v in st.items() if k.startswith("dup_vs_")}
        out += [
            f"- **Generation**: {st['raw']} matura-style items (MC with distractors, true/false, matching, multi-select, short answer, "
            "numeric) written from the articles in the exact target answer format.",
            _verification_sentence(st),
            f"- **Leakage control (embedding + fuzzy dedup, bge-m3 on question+answer)**: dropped "
            + ", ".join(f"{n} vs {k.removeprefix('dup_vs_')}" for k, n in protected.items())
            + f", {st.get('intra_dup', 0)} internal near-duplicates. The check provably ran: counts and examples are in "
            "`data/synthetic/dedup_report.json`, and training refuses a clean set deduplicated against different eval files.",
            f"- **Kept**: {st['kept']} items; MC answer letters "
            + ", ".join(f"{k} {v}" for k, v in report.get("mc_answer_letters", {}).items()) + " (options re-lettered at render time).",
        ]
    audit = _load_json(ROOT / "data/synthetic/key_audit.json")
    if audit:
        out.append(f"- **Independent key audit**: a separate blind solver re-answered a random {audit['n_sampled']}-item sample of the final "
                   f"set: {audit['counts'].get('key_wrong', 0)} wrong keys (95% upper bound {100 * audit['key_wrong_rate_wilson95_upper']:.1f}%), "
                   f"{audit['counts'].get('ambiguous', 0)} ambiguous, {audit['counts'].get('trivia', 0)} correct but low matura relevance.")
    if pool:
        out.append(f"- **General replay**: {len(pool)}-row Polish instruction pool (openeurollm/EU-Instruct-Synthetic `pl`) mixed in to "
                   "guard general capability; answers capped so matura tokens keep a meaningful share of the loss.")
    out.append("")
    return out


def _variant_table(summary: dict) -> list[str]:
    lines = ["| set / variant | points | score% | parse-fail% | lenient points | loglik acc |", "|---|---|---|---|---|---|"]
    for set_name, s in summary.get("sets", {}).items():
        ll = s.get("loglik") or {}
        for v, vs in s.get("variants", {}).items():
            lines.append(f"| {set_name} / {v} | {vs['points']:.0f}/{vs['max_points']:.0f} | {vs['score_pct']:.1f} | "
                         f"{100 * vs['parse_fail_rate']:.1f} | {vs['points_lenient']:.0f} | "
                         f"{_fmt(None if ll.get('acc') is None else 100 * ll['acc'], 1)} |")
    return lines


def section_baseline(rdir: pathlib.Path, base_model: str) -> list[str]:
    summary = _load_json(rdir / "runs" / "base" / "summary.json")
    out = ["## 3. Baseline: untouched base model", ""]
    if not summary or not summary.get("is_base", True):
        return out + [f"_Not evaluated yet: `python scripts/run_eval.py --model {base_model} --base`._", ""]
    h = summary["headline"]
    gen = summary.get("general_heldout") or {}
    out += [
        f"`{base_model}` scores **{h['points']:.0f}/{h['max_points']:.0f} ({h['score_pct']:.1f}%)** strict on the held-out set, "
        f"parse-fail rate {100 * h['parse_fail_rate']:.1f}%, lenient {h['points_lenient']:.0f} points "
        f"(→ {h['points_lenient'] - h['points']:.0f} points lost purely to answer format). General NLL {_fmt(gen.get('nll'), 3)}. "
        f"Evaluated on {summary.get('accelerator') or summary.get('device')} ({summary.get('dtype')}).",
        "",
        *_variant_table(summary),
        "",
    ]
    return out


def _latest_runs(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in rows:
        if r.get("event", "run") == "run":
            latest[r["run_name"]] = r
    return sorted(latest.values(), key=lambda r: r.get("timestamp", ""))


def section_runs(rdir: pathlib.Path) -> list[str]:
    runs = _latest_runs(_rows(rdir / "runs.jsonl"))
    out = ["## 4. Experiments (chronological; latest record per run)", ""]
    if not runs:
        return out + ["_No fine-tuning runs recorded yet: `python scripts/train.py --config configs/run1.yaml`._", ""]
    candidate = (_load_json(rdir / "CANDIDATE.json") or {}).get("run_name")
    out += [
        "Δ = strict points over base on held-out/canonical. **format** = Δstrict − Δlenient (points gained by answering in a parseable "
        "form), **knowledge** = Δlenient (points gained regardless of format). **gen-heldout loss**: completion loss on the "
        "independent general Polish set before training (step 0 = base) → after the last epoch. ★ = current candidate.",
        "",
        "| run | hypothesis / notes | lr | r | ep | gen ratio | Δ pts [CI95] | format | knowledge | parse-fail% | secondary Δpp | general NLL Δ% | gen-heldout loss | gate | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        hp = r.get("hparams") or {}
        d, dl = r.get("delta_points"), r.get("delta_lenient_points")
        fmt_gain = None if d is None or dl is None else d - dl
        ci = r.get("ci95") or [None, None]
        sec = r.get("secondary_delta_score_pct") or {}
        tr = r.get("train") or {}
        step0 = (tr.get("step0") or {}).get("general_heldout_loss")
        last = (tr.get("last_epoch") or {}).get("general_heldout_loss")
        gate = "—" if r.get("gate_passed") is None else ("pass" if r["gate_passed"] else "FAIL " + ",".join(r.get("gate_failed_rules") or []))
        status = "★" if r["run_name"] == candidate else ("demoted" if r.get("demoted") else ("promoted earlier" if r.get("promoted") else ""))
        notes = _cell(" ".join(str(r.get("notes") or "").split())[:140]) or "—"
        out.append(
            f"| {_cell(r['run_name'])} | {notes} | {_v(hp.get('lr'))} | {_v(hp.get('r'))} | {_v(hp.get('epochs'))} | "
            f"{_v(hp.get('general_ratio'))} | {_fmt(d, 1, True)} [{_fmt(ci[0], 1, True)}, {_fmt(ci[1], 1, True)}] | "
            f"{_fmt(fmt_gain, 1, True)} | {_fmt(dl, 1, True)} | "
            f"{_fmt(None if r.get('parse_fail_rate') is None else 100 * r['parse_fail_rate'], 1)} | "
            f"{'; '.join(f'{_cell(k)} {_fmt(v, 1, True)}' for k, v in sec.items()) or '—'} | "
            f"{_fmt(None if r.get('general_nll_rel_change') is None else 100 * r['general_nll_rel_change'], 2, True)} | "
            f"{_fmt(step0, 3)} → {_fmt(last, 3)} | {gate} | {status} |"
        )
    out.append("")
    passed = [r for r in runs if r.get("gate_passed")]
    failed = [r for r in runs if not r.get("gate_passed")]
    if passed:
        best = max(passed, key=lambda r: r.get("delta_points") or 0)
        d, dl = best.get("delta_points") or 0, best.get("delta_lenient_points") or 0
        share = "mostly answer-format compliance" if d - dl > dl else "mostly knowledge (format-independent)"
        out.append(f"**What worked**: `{best['run_name']}` is the best gate-passing run at {_fmt(d, 1, True)} points "
                   f"({share}: format {_fmt(d - dl, 1, True)}, knowledge {_fmt(dl, 1, True)}).")
    for r in failed:
        reasons = [x for x in (r.get("gate_reasons") or []) if x.startswith("FAIL")]
        out.append(f"- `{r['run_name']}` rejected by the gate: " + "; ".join(reasons[:3]))
    out.append("")
    return out


def section_candidate(rdir: pathlib.Path) -> list[str]:
    cand = _load_json(rdir / "CANDIDATE.json")
    tags = _load_json(rdir / "tags.json") or {}
    out = ["## 5. Current submission candidate", ""]
    if not cand:
        return out + ["_No registry yet._", ""]
    out.append(f"**{cand['run_name']}** (`{cand['model_path']}`), Δ {_fmt(cand.get('delta_points'), 1, True)} points, "
               f"score {_fmt(cand.get('score_pct'), 1)}%. Reason: {_v(cand.get('reason'))}. Tags: {', '.join(cand.get('tags') or []) or '—'}.")
    if tags:
        out.append("")
        out.append("Tags: " + ", ".join(f"`{t}` → {r}" for t, r in sorted(tags.items())))
    out += ["", "The untouched base model remains the standing fallback: a run replaces it only by passing every gate rule "
            "(strict delta, no parse-fail increase, general NLL within 5%, no regression on other prompt variants, the real-exam set, "
            "log-likelihood accuracy or lenient score) and beating the current candidate.", ""]
    return out


def build(cfg: dict) -> str:
    rdir = results_dir(cfg)
    base_model = cfg["model"]["base"]
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    lines = [
        f"# Fine-tuning {base_model} for matura-style exams — write-up",
        "",
        f"_Generated {now} by `scripts/writeup.py` from committed data and `{rdir.relative_to(ROOT) if rdir.is_relative_to(ROOT) else rdir}`._",
        "",
        "**Approach in one line:** treat answer-format compliance as a training target, measure everything as delta over the untouched "
        "base under a fingerprinted strict grader, and only ever promote a model that passes a no-regression gate.",
        "",
    ]
    for sec in (section_eval(), section_data(), section_baseline(rdir, base_model), section_runs(rdir), section_candidate(rdir)):
        lines += sec
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml", help="selects the base model (results/<slug>) and paths")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override, e.g. paths.results_root=.smoke/results")
    ap.add_argument("--out", help="default: results/<slug>/WRITEUP.md")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    text = build(cfg)
    if args.stdout:
        print(text)
        return 0
    out = resolve_path(args.out) if args.out else results_dir(cfg) / "WRITEUP.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
