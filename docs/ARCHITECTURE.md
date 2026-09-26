# Architecture & module contracts

Binding contract between modules. If you change an interface, update this file.
Verified library facts live in `docs/research/*.md` (read `hf_stack_api.md` GOTCHAS before touching HF code).

## Principles (non-negotiable)

1. **One format definition.** `eval/answer_format.py` owns instruction text, target completions and parsers.
   Training rows (`eval/prompts.build_training_example`) and grading import from it. Never re-implement.
2. **Strict grading is the headline number.** Lenient is diagnostic (`format_loss_points = lenient - strict`).
3. **Delta-over-base is the primary metric.** Every non-base eval is compared to the base model's eval with an
   identical *eval fingerprint*. Mismatches are refused unless `--allow-fingerprint-mismatch` (recorded).
4. **Never be without a candidate.** `results/<base_slug>/CANDIDATE.json` always exists; it starts as the untouched
   base model and changes only via the promotion gate (`train/registry.py`), demotion to base when the candidate fails
   re-evaluation, or an explicit logged `promote.py --force`.
5. **No leakage.** Synthetic items are deduplicated (embedding + fuzzy) against held-out eval, dev, external eval
   and the external-exam blocklist. Dropped counts + examples are logged to `data/synthetic/dedup_report.json`.
6. **Teammate UX.** `pip install -r requirements.txt` (+ `hf auth login` & accept the gated Bielik license), then
   `python scripts/train.py --config configs/run1.yaml` and `python scripts/run_eval.py --model checkpoints/run1`.
   All data needed for training/eval is committed under `data/`, except the CKE-derived external eval and
   blocklist files (gitignored: `python scripts/fetch_external.py --only ext,blocklist` re-creates them) and the
   organizers' prawko-v2 eval splits (gitignored: `python scripts/import_prawko.py` re-creates them).

## Layout

```
configs/   base.yaml (defaults, implicitly inherited) | run1.yaml (v0-safe) | smoke.yaml (tiny model, local) | sweep.yaml | bielik11b.yaml
data/
  eval/heldout.jsonl            60 authored + blind-verified matura-style items (70 pts)   NEVER train
  eval/dev.jsonl                20 more verified items for prompt/debug iteration             NEVER train
  eval/verification_log.jsonl   per-item author/solver/critic verdicts (write-up evidence)
  eval/ext_llmzszl_matura.jsonl 374 real CKE matura MC items from amu-cai/llmzszl-dataset     NEVER train (secondary eval)
                                (377 source rows - 2 exact duplicates - 1 row with a merged/empty option)
  eval/ext_prawko_dev.jsonl     25 official Polish driving-licence MC items, prawko-v2 dev     NEVER train
  eval/ext_prawko_test.jsonl    40 more, prawko-v2 test (the organizers' own held-out exam)    NEVER train
                                (gitignored third-party rows: re-create with scripts/import_prawko.py; option text
                                 verbatim, subject "wos", points 1, source "prawko-v2:<split>". Scored ONLY under the
                                 organizers' protocol: `variants: []`; `optional: true`, so a missing file is skipped)
  eval/general_heldout.jsonl    300 human-annotated Polish instruction pairs, NASK-PIB/PLLuM-Align (cc-by-sa-4.0)
  blocklist/*.jsonl             real exam items from public datasets (dedup targets only)     NEVER train
  wiki/titles.yaml, wiki/articles.jsonl, wiki/fetch_report.json
  synthetic/raw/<shard>.jsonl       generated items (schema: eval/schema.py; source=wikipedia:<title>)
  synthetic/verify/<shard>.jsonl    open-book blind-solver answers {id, payload, confidence, concern}
  synthetic/clean.jsonl             validated + verified + deduplicated items
  synthetic/dedup_report.json       counts + examples of every drop reason, verify coverage per shard, sha256-16 of every protected file
  synthetic/key_audit.json          independent blind audit of a seeded 64-item sample of clean.jsonl (wrong-key rate + Wilson bound)
  general/train_pool.jsonl      Polish instruction pairs, openeurollm/EU-Instruct-Synthetic `pl` (apache-2.0), filtered
  processed/<run_name>/         rendered TRL rows (gitignored, rebuilt per run from config)
eval/      schema, answer_format, prompts, grader, modeling (contract, DONE) + generation, loglik, general_loss, runner, stats, compare
train/     config, wiki, synth_prompt (contract, DONE) + external, dedup, synth_verify, dataset, generate_synthetic, sft, merge, registry, pipeline
scripts/   fetch_wiki, show_items, synth_task (DONE) + run_eval, compare, status, promote, fetch_external, import_prawko, build_data, generate_synthetic, train, merge_and_export, sweep, writeup
results/<base_slug>/  runs.jsonl | CANDIDATE.json | tags.json | LEADERBOARD.md | WRITEUP.md (scripts/writeup.py) | runs/<run_name>/{summary.json, compare.json, predictions.*.jsonl, loglik.*.jsonl}
                      sweeps/<sweep_name>/{<run>.log, sweep_summary.json}
checkpoints/<run_name>/  merged eval-ready model (root) + adapter/ (LoRA) + train_metrics.json + resolved_config.yaml + training_meta.json   (gitignored)
                         (pipeline.keep_merged=candidate_only deletes the merged weights of non-candidate runs; adapter/ is kept)
                         shared by EVERY base model: training refuses a run name that is any base model's candidate checkpoint
                         or whose training_meta.json names another base model, so run names must be unique across base models
checkpoints/CANDIDATE-<base_slug> -> <run_name>   per-base-model symlink to that base model's candidate checkpoint (absent while its candidate is the base model)
tests/     pytest (fast; no network; tiny model tests marked @pytest.mark.slow). `pytest` runs the fast suite, `pytest -m slow` the model tests
```

`base_slug` = last path component of `model.base`, lowercased (e.g. `bielik-4.5b-v3.0-instruct`). Paths are
configurable: `paths.results_root` (default `results`), `paths.checkpoints_root` (default `checkpoints`),
`paths.processed_root` (default `data/processed`). Smoke runs use `.smoke/` roots so they never touch real results.

Every script starts with the sys.path bootstrap (repo root must precede `scripts/` so `import train` is the package):
```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
```

## Data contracts

**Item** — `eval/schema.py` docstring. Same schema for eval sets, external eval, blocklist and synthetic items
(blocklist items may lack `answer`; load them with `validate=False`).

**General instruction pair** (`data/general/train_pool.jsonl`, `data/eval/general_heldout.jsonl`):
`{"id": str, "source": str, "messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`
(optional leading system message; last message is the assistant response; loss/NLL only on it).

**Processed training row** (TRL conversational prompt-completion; verified completion-only loss incl. `<|im_end|>`):
`{"prompt": [messages...], "completion": [{"role": "assistant", "content": ...}], "kind": "matura"|"general", "id": ..., "variant": ...|null}`
Extra columns (`kind`, `id`, `variant`) are removed before passing to SFTTrainer.

## Config keys (see configs/base.yaml for defaults)

`run_name, tags, seed, notes, model.{base, quantization, dtype, attn_implementation, max_length}, lora.{r, alpha, dropout, target_modules, use_rslora},
data.{synthetic, general_pool, general_heldout, n_matura, general_ratio, general_max_completion_chars, val_fraction, variant_mix, system_prompt_prob, system_prompts, shuffle_options, renders_per_item, general_val_n, general_heldout_n},
train.{epochs, learning_rate, lr_scheduler, warmup_ratio, per_device_batch_size, grad_accum, weight_decay, max_grad_norm, gradient_checkpointing, eval_steps, logging_steps, max_steps, packing},
eval.{sets[{path, name?, variants, loglik, optional, organizer?, organizer_rotate?}], primary_set, primary_variant, max_new_tokens, max_new_tokens_reasoning, batch_size, quantization, system_prompt, general_loss, general_loss_max_items, limit},
pipeline.{merge_after_train, auto_eval, keep_merged},
gate.{min_delta_points, max_parse_fail_rate_increase, max_general_nll_increase, max_variant_regression_points, max_secondary_score_drop_pct, min_net_score_pct, max_loglik_acc_drop_pct, must_beat_current_candidate, min_improvement_points, max_lenient_regression_points, max_organizer_acc_drop_pct},
paths.{results_root, checkpoints_root, processed_root}`

`load_config` rejects any key of a config file or `--set` override that configs/base.yaml does not define (ValueError with a
did-you-mean hint; `run_name` and `data.variant_mix.<variant>` are free). `variant_mix` is replaced, not merged: a run config's
table is the whole mix (a variant it leaves out gets weight 0), exactly like `--set data.variant_mix={...}`.
`general_ratio` counts ROWS; the loss is token-weighted, so `general_max_completion_chars` caps general answers (train and
val_general rows) and `train_metrics.loss_tokens.matura_share` logs the real split.

**Organizer protocol (eval-only, additive).** An eval set spec may set `organizer: true`: its `mc` items are also
scored the way the organizers' reference grader works (`eval/organizers.py` — chat template, logits at the FIRST answer
position, softmax over the single-token option letters only, argmax; no generated text is read, so answer-format
compliance earns nothing there), and `organizer_rotate: true` repeats it with the options rotated by one (letter-position
bias probe). Non-`mc` items are skipped. It adds numbers, never replaces any: strict `Odpowiedź: X` grading stays the
headline. `gate.max_organizer_acc_drop_pct` (3.0 pp) is gate rule 9. configs/base.yaml enables it on `heldout`,
`ext_llmzszl_matura` and the two organizer-only prawko sets (`variants: []`, `loglik: false`).

TRL/transformers mapping (verified, transformers 5.17 / trl 1.13): `warmup_ratio` -> `SFTConfig(warmup_steps=<float ratio>)`;
`max_length` -> `SFTConfig(max_length=...)`; `eval_strategy="epoch"` when `train.eval_steps` is null; eval_dataset is a dict
`{"matura": ..., "general": ..., "general_heldout": ...}` -> metrics `eval_matura_loss`, `eval_general_loss`, `eval_general_heldout_loss`.
Pass a model object + `peft_config` (never a PeftModel + peft_config). Only call `prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})` for 4-bit.
`dataloader_pin_memory=False` on MPS. Call `trainer.evaluate()` BEFORE `trainer.train()` to record step-0 (== base) losses.
Filter rows whose tokenized prompt+completion exceed `max_length` BEFORE training (TRL silently drops fully masked rows) and log the count.

## Function signatures (code against these)

```python
# eval/modeling.py (DONE)
load_model(path_or_id, *, dtype="auto", quantization="none", device=None, attn_implementation="auto", for_training=False, merge_adapter=True)
load_tokenizer(path_or_id, padding_side="left");  load_model_and_tokenizer(path, **kw);  detect_device();  resolve_dtype();  resolve_quantization();  free_model(model)
is_adapter_dir(path);  adapter_base_model(path);  sanitize_generation_config(model);  resolve_attn_implementation(name)   # auto -> sdpa
    # resolve_dtype auto on CUDA: bf16 only on GPUs with native bf16 (compute capability >= 8; T4/V100 get fp16).
    # load_model warns (RuntimeWarning) when device_map="auto" offloaded modules to CPU/disk; model._matura_load_info records
    # {device, dtype, quantization, attn_implementation (actual), source, base_source, adapter}.

# eval/generation.py
generate_responses(model, tokenizer, messages_list: list[list[dict]], *, max_new_tokens: int, batch_size: int, desc: str | None = None) -> list[str]
    # render with apply_chat_template(tokenize=False, add_generation_prompt=True), tokenize with add_special_tokens=False (Bielik double-BOS gotcha),
    # left padding, greedy do_sample=False, decode only new tokens with skip_special_tokens=True. Sort by length for batching, restore order.
    # eos ids = tokenizer eos + model.generation_config eos (Bielik: <|im_end|>, </s>); desc labels an optional progress bar.

# eval/loglik.py
LLMZSZL_TEMPLATE  # "Przykładowe pytanie egzaminacyjne, test jednokrotnego wyboru\n\n{question}\n{options}\nPrawidłowa odpowiedź:"
score_mc_loglik(model, tokenizer, items: list[dict], *, batch_size: int) -> list[dict]   # mc items only; raw prompt (NO chat template, lm-eval style);
    # continuation " A"/" B"/...; record {id, subject, gold, pred, correct, logprobs:{A:..}}
    # context ids = leading_special_ids(tokenizer) (BOS, never a trailing EOS) + tokenize(prompt, add_special_tokens=False);
    # continuation ids = suffix of tokenize(prompt + " A") (encode_continuation), falling back to tokenize(" A") when the boundary merges.
    # lm-eval's HF model adds no BOS by default, so absolute accuracy may differ from published LLMzSzŁ numbers (deltas are unaffected).
summarize_loglik(records) -> {"n", "acc", "per_subject": {s: {"n", "acc"}}}

# eval/general_loss.py
general_nll(model, tokenizer, pairs: list[dict], *, batch_size: int, max_length: int) -> {"nll": float (mean per assistant token), "tokens": int, "n": int, "skipped": int}
    # tokenized like TRL prompt-completion rows (response includes <|im_end|>); skipped = pairs not scorable within max_length

# eval/runner.py
eval_sets_from_cfg(cfg) -> list[dict]      # normalized [{name, path, variants, loglik, optional}], name defaults to file stem; variants default
                                           # [eval.primary_variant], loglik default false; missing optional sets skipped (warning), duplicate names /
                                           # unknown variants / missing required sets raise
eval_fingerprint(cfg, load_info) -> (str, dict)   # sha256-16 over: FORMAT_VERSION, eval_code_sha256 (sha256-16 of eval/{answer_format,grader,
                                                 # prompts,schema,generation,loglik,general_loss}.py: grading/prompt/tokenization code), per-set (name, file
                                                 # sha256, variants, loglik), max_new_tokens, max_new_tokens_reasoning (only when a set uses
                                                 # reason_then_answer), loglik_template_sha256 (only when a set has loglik), system_prompt, limit,
                                                 # eval.batch_size, general_loss (heldout file sha256, max_items, model.max_length), dtype, quantization,
                                                 # device, attn_implementation (resolved), torch, transformers, accelerator (CUDA device name, else device)
planned_load_info(cfg) / model_load_info(model) -> {device, dtype, quantization, attn_implementation, torch, transformers, accelerator}
# results_dir / run_dir / checkpoint_dir / processed_dir / base_slug / is_base_run_name / candidate_link_name live in train/config.py (DONE)
run_eval(model_path, run_name, cfg, *, is_base=False, limit=None, model=None, tokenizer=None, force=False) -> dict(summary)
    # if summary.json exists with same fingerprint, same model identity and same model_signature (adapter file hashes plus the merged
    # weight file names/sizes/mtimes at the dir root when present) and not force -> return cached. Otherwise deletes the run dir's stale
    # summary/compare/predictions/loglik first. eval.limit also caps the general-NLL pairs. Loads with eval.quantization (default none).
    # A checkpoint dir whose merged weights were cleaned up is evaluated from its adapter/ (base + adapter merged in memory).
    # Base run names (base, base-limit<N>, base-dbg<fp8>) require is_base=True (ValueError). A file lock (runs/<name>/.eval.lock) serializes
    # evals of the same run dir across processes: the second one reuses the first one's fresh summary. predictions/loglik files are atomic.
load_summary(run_dir) -> dict | None
checkpoint_eval_overrides(model_path) -> list[str]   # --set style leaves of eval.*, model.{dtype,attn_implementation,max_length},
    # data.general_heldout from <checkpoint>/resolved_config.yaml ([] when absent): run_eval.py / compare.py mode B without --config
debug_suffix(cfg, plain_cfg, *, limit) -> str        # "-limit<N>" for CLI --limit; "-dbg<fingerprint[:8]>" when CLI overrides changed the fingerprint

# eval/stats.py
paired_bootstrap_ci(base_points: list[float], run_points: list[float], n_boot=10000, seed=0, alpha=0.05) -> (lo, hi)
sign_test(wins: int, losses: int) -> float  # two-sided exact binomial p-value

# eval/compare.py
compare_runs(base_dir: Path, run_dir: Path, *, allow_fingerprint_mismatch=False) -> dict   # schema below; refuses a base_dir whose summary
    # is not a base-model eval (is_base must be true)
    # sets.<set>.organizer is added only when BOTH summaries carry that set's organizer block (absent, never null)
write_compare(run_dir: Path, base_dir: Path, **kw) -> dict   # writes run_dir/compare.json
format_compare_table(cmp: dict) -> str   # + an "organizer (first-token letter argmax)" section when organizer deltas exist
fingerprint_diff(base_summary: dict, run_summary: dict) -> {dotted.key: {"base": ..., "run": ...}}   # names the differing fingerprint parts

# train/external.py (+ scripts/fetch_external.py)
build_ext_llmzszl_matura(out_path) -> int;  build_blocklist(out_dir) -> dict[str, int];  build_general_pool(out_path, n, seed) -> int;  build_general_heldout(out_path, n, seed) -> int

# scripts/import_prawko.py   (the organizers' prawko-v2 driving-licence questions -> our item schema)
fetch_data(url=DATA_URL, from_file=None, *, timeout=60) -> {split: rows}   # downloads their data.json (requests) or reads a local copy
convert_row(row, split) -> item;  convert_split(rows, split) -> list[item]   # answer index -> letter, option text VERBATIM, points 1,
    # id "prawko-<their id>", subject "wos", source "prawko-v2:<split>"; validated with eval.schema.validate_item
answer_counts(items) -> {letter: n};  source_points(rows) -> {their points value: n rows}   # printed only: their harness is unweighted
    # writes data/eval/ext_prawko_{dev,test}.jsonl (25 / 40 rows, gitignored). Their `train` split is never imported.

# train/dedup.py
DedupConfig(model="BAAI/bge-m3", thr_qa=0.80, thr_q=0.90, thr_fuzzy=92, thr_intra_qa=0.95, batch_size=64,
            min_fuzzy_words=6, fuzzy_min_len_ratio=0.5, max_seq_length=512, chunk_size=1024, n_examples=20, cache_dir="data/synthetic/.emb_cache")
dedup_items(candidates: list[dict], protected: dict[str, list[dict]], cfg: DedupConfig | None = None, embed_fn=None) -> (kept: list[dict], report: dict)
    # embed_fn(texts) -> L2-normalizable ndarray; None loads cfg.model (sentence-transformers) and releases it afterwards
    # protected = {"heldout": [...], "dev": [...], "ext_llmzszl_matura": [...], "blocklist": [...]}
    # embed "stem (options and exam instructions removed) + '\nOdpowiedź: ' + gold answer TEXT" (QA) and stem alone (Q); flag if cos_QA>=thr_qa or cos_Q>=thr_q
    # or rapidfuzz token_set_ratio(stem, stem)>=thr_fuzzy; items without answers (blocklist) use Q + fuzzy only; then intra-synthetic dedup (cos_QA>=thr_intra_qa keep first).
    # Exam instructions ("Dokończ zdanie. Wybierz właściwą odpowiedź spośród podanych.", "Oceń prawdziwość…") are stripped from both stems first.
    # fuzzy guard: both stems need >= min_fuzzy_words content words (>= 2 chars incl. a letter) and token counts within fuzzy_min_len_ratio
    # (token_set_ratio is 100 whenever one token set contains the other, so short stems would match any long passage).
    # report: counts per reason & per protected set, threshold values, 20 examples per reason (candidate text, matched text, scores).

# train/synth_verify.py
verify_filter(items: list[dict], answers: dict[str, dict], *, drop_concerns: bool = True) -> (kept, report)   # drop: unverified | solver_disagrees | low_confidence | solver_concern (agreeing solver still flagged a concern; scripts/build_data.py --keep-concerns keeps them)
    # payload names exactly ONE answer: "B lub C", "3,5 albo 7", a 2-element list for mc/short/numeric disagree) and not (confidence=="low")
# scripts/build_data.py: validate -> verify_filter -> dedup_items -> clean.jsonl + dedup_report.json (inputs.protected_files {path: sha256-16} of
#   heldout/dev/ext/blocklist/--extra-protected files; inputs.verify_coverage per raw shard). Exits 1 when a non-skipped raw shard has verify
#   coverage below --min-verify-coverage (default 0.9) unless --allow-missing-verify.

# train/dataset.py
build_processed(cfg, out_dir=None, *, allow_stale_dedup=False) -> {"train": Path, "val_matura": Path, "val_general": Path, "general_heldout": Path, "stats": dict}
    # deterministic from cfg.seed; split matura items into train/val BEFORE rendering, by whole source article (items sharing `source` never
    # straddle train/val); per train item sample variant from variant_mix (reason_then_answer falls back to canonical when no rationale),
    # optional system prompt, option shuffling with answer remap; general rows = round(n_train_matura * r / (1 - r)) (rows, before sft's
    # length filter: stats.general_ratio_built); pool rows with an answer longer than data.general_max_completion_chars are not used;
    # val_general disjoint from train general rows.
    # PROTECTED_EVAL_FILES = heldout, dev, ext_llmzszl_matura, ext_prawko_dev, ext_prawko_test (adding an eval set means
    # rerunning scripts/build_data.py --extra-protected <file>, otherwise the staleness check below refuses the build)
    # raises LeakageError if an item id OR identical rendered task belongs to heldout/dev/ext/prawko or any eval.sets file, or when
    # dedup_report.json inputs.protected_files hashes differ from the current heldout/dev/ext/eval.sets/data/blocklist files (or miss one;
    # skipped when the report or its hashes are absent; allow_stale_dedup=True only warns); pool rows whose prompt is anywhere in the
    # data.general_heldout FILE are dropped; options that refer to each other by letter/position, and mc/multi/match items whose question,
    # context or left column refers to options by letter ("tylko A", "A–C", "odpowiedź A"), are never re-lettered (nor renders whose rationale
    # cites letters); a missing data.synthetic raises FileNotFoundError pointing to scripts/build_data.py.

# train/sft.py
train_model(cfg, data: dict) -> {"adapter_dir": Path, "train_metrics": dict}
    # train_metrics: {"step0": {"matura_loss", "general_loss", "general_heldout_loss"}, "epochs": [{"epoch", "step", "train_loss", "matura_loss", "general_loss", "general_heldout_loss"}],
    #                 "dropped_too_long": int (train split only; eval-split drops are in dropped_too_long_by_split), "dropped_too_long_by_split": {split: int},
    #                 "dropped_too_long_by_kind": {kind: int} (train split), "general_ratio_trained": float (after the length filter),
    #                 "loss_tokens": {"matura": int, "general": int, "matura_share": float} (completion tokens of the kept train rows),
    #                 "n_train": int, "n_train_prepared": int, "n_eval": {split: int},
    #                 "global_steps": int, "train_loss_mean": float, "runtime_s": float, "device", "quantization", "dtype"}
    #   (+ "stage_seconds" added by run_pipeline). Removes an existing checkpoints/<run_name>/ first (never a candidate's).
    # The torch RNG is seeded with cfg.seed right before SFTTrainer creates the LoRA layers (identical configs start from identical adapters).
    # also writes <checkpoint>/{train_metrics.json, resolved_config.yaml, training_meta.json}
assert_trainable_run_name(cfg)   # refuses base run names, any base model's candidate checkpoint (every results/*/CANDIDATE.json and
    # CANDIDATE* link target) and a checkpoint whose training_meta.json names another base model

# train/merge.py (+ scripts/merge_and_export.py)
merge_and_export(adapter_dir, out_dir, *, base_model=None, dtype="bf16", device="auto") -> Path   # unquantized base; merged dir must NOT contain
    # adapter_config.json; device auto = cuda if available else cpu; copies train_metrics/resolved_config/training_meta next to the weights;
    # refuses an out_dir whose adapter/ is a different adapter (another run's checkpoint, or a CANDIDATE link to one)

# train/registry.py
ensure_candidate(cfg) -> dict   # also called by scripts/run_eval.py after a full --base eval, so CANDIDATE.json exists right after the baseline
evaluate_gate(cmp: dict, candidate: dict | None, gate_cfg: dict, *, required_scope: dict | None = None) -> (passed: bool, reasons: list[str])
    # candidate None = base; reasons are "PASS [rule] ..." / "FAIL [rule] ..." lines (failed_rules(reasons) -> ["7", "limit", "5b", ...]);
    # required_scope = reference_eval_scope() adds the [scope] check (record_run and compare.py's dry run always pass it)
reference_eval_scope() -> {set_name: {"variants", "loglik"}}   # configs/base.yaml eval.sets (optional sets only when their file exists)
record_run(cfg, *, run_dir, cmp, train_metrics, model_path, adapter_path) -> dict(row)   # gate -> promote (or demote) -> tags -> runs.jsonl, CANDIDATE.json, LEADERBOARD.md, CANDIDATE link, cleanup
    # ValueError (nothing written) when run_dir's run name is the candidate's but model_path is another checkpoint/model (its adapter/ counts as
    # the same checkpoint); a promoted candidate gets "warnings" (e.g. no merged model: rebuild command)
check_run_name_owner(cfg, run_name, model_path)   # the same ValueError as a preflight (run_eval.py, compare.py mode B call it before evaluating)
promote(cfg_or_results_dir, run_name, *, reason, tags=(), force=False) -> dict
render_leaderboard(results_dir) -> str
cleanup_checkpoints(results_dir, checkpoints_root, policy) -> list[str]   # candidate_only: delete merged weights of non-candidate runs, keep adapter/
    # (checkpoints without adapter/ are never touched: the merged model would be the only copy)
protected_checkpoints(results_dir, checkpoints_root) -> set[str]   # every base slug's candidate + every CANDIDATE* link target (never cleaned/overwritten)
candidate_link(checkpoints_root, results_dir) -> Path             # <checkpoints_root>/CANDIDATE-<base_slug>
load_runs(results_dir);  load_candidate(results_dir);  load_tags(results_dir);  leaderboard_rows(results_dir);  format_leaderboard_table(rows)

# train/pipeline.py (+ scripts/train.py)
run_pipeline(cfg, *, do_merge=True, do_eval=True) -> dict   # build_processed -> train_model -> merge -> ensure base eval -> run_eval -> write_compare
    #   -> refresh_stale_candidate -> record_run
    # preflight before training: ensure_candidate, assert_trainable_run_name, eval_sets_from_cfg + check_eval_vram (when evaluating: on CUDA with
    # eval.quantization none, refuses when the base's bf16 weights exceed 80% of GPU memory; use eval.quantization=4bit or a bigger GPU).
    # refresh_stale_candidate(cfg, fingerprint): a non-base candidate recorded under another eval fingerprint is re-evaluated and re-recorded under
    # the current eval first (skipped with a notice when its checkpoint is not on this machine; gate rule 7 then fails).
    # Evaluating without merging is refused (the gate must judge the merged export that would be shipped).

# scripts/sweep.py   sweep file: {name, base_config, budget_hours, runs: [{name, set: {dotted.key: value}}], grid?: {dotted.key: [values]}}
    # one scripts/train.py subprocess per run (grid runs named g-<key>=<value>_...); runs already in runs.jsonl with the same config_hash are
    # skipped, a run name recorded with a different config_hash makes the sweep refuse to start (rename edited runs); tags=[<sweep name>]
    # unless set; a run is launched while the remaining budget >= mean duration of this sweep's successful runs (--est-hours before the first).
    # outputs: results/<slug>/sweeps/<name>/{<run>.log, sweep_summary.json}
```

## Eval summary (`results/<slug>/runs/<run_name>/summary.json`)

```json
{"run_name": "run1", "model": "checkpoints/run1", "model_source": "checkpoints/run1", "model_signature": "9f3c…",
 "is_base": false, "base_model": "speakleash/Bielik-4.5B-v3.0-Instruct",
 "timestamp": "ISO-8601", "fingerprint": "ab12…", "fingerprint_parts": {…}, "device": "cuda", "dtype": "bfloat16", "quantization": "none",
 "attn_implementation": "sdpa", "torch": "2.14.0", "transformers": "5.17.0", "accelerator": "NVIDIA A100-SXM4-40GB",
 "primary_set": "heldout", "primary_variant": "canonical",
 "headline": {"points": 41.0, "max_points": 70.0, "score_pct": 58.57, "parse_fail_rate": 0.0, "points_lenient": 42.0},
 "sets": {"heldout": {"path": "data/eval/heldout.jsonl", "n_items": 60,
                      "variants": {"canonical": <eval.grader.summarize()>, "bare": {…}, "payload_only": {…}},
                      "loglik": {"n": 27, "acc": 0.63, "per_subject": {…}},
                      "organizer": {<eval.organizers.summarize_organizer(): n, correct, accuracy,
                                     mean_correct_probability, mean_abc_mass, per_subject>,
                                    "skipped": 33, "rotated": <same keys> | null}},
          "ext_llmzszl_matura": {…}},
 "general_heldout": {"nll": 1.234, "tokens": 45678, "n": 300, "skipped": 0} | null,
 "elapsed_s": 321.0}
```
Files: `predictions.<set>.<variant>.jsonl` (grader records + `prompt`), `loglik.<set>.jsonl`, `organizer.<set>.jsonl`
(every organizer record, the `rotate: 0` and `rotate: 1` rows in one file). The `organizer` block is present only for
sets whose spec sets `organizer: true`; `skipped` counts items the protocol cannot score (non-`mc`, or letters that are
not single tokens), and `rotated` is null unless `organizer_rotate: true`.
`model` is the model identity (hub id, or repo-relative resolved path); `model_source` is what was actually loaded (e.g. `…/adapter` when the
merged weights were cleaned up); `model_signature` invalidates the cache when the weights behind the same path change.
Debug evals never overwrite full evals (`scripts/run_eval.py`, `scripts/compare.py --model`): CLI `--limit N` stores `runs/<name>-limit<N>`
(compared with `runs/base-limit<N>`), and any CLI override that changes the eval fingerprint of the same config without it (`--variants`,
`--sets`, `--set eval.*` / `model.*`) stores `runs/<name>-dbg<fingerprint[:8]>` (with `runs/base-dbg<…>`; both suffixes may combine). A
config-FILE `eval.limit` (e.g. configs/smoke.yaml) keeps the plain names. `base` names are reserved for base-model evals. Without `--config`
both scripts evaluate a checkpoint with its `resolved_config.yaml` eval settings (checkpoint_eval_overrides), so pipeline evals are reused.

## Compare (`compare.json`)

```json
{"run_name": "run1", "base_run": "base", "fingerprint": "ab12…", "fingerprint_match": true, "fingerprint_diff": {},
 "eval_limit": null, "eval_sets": [{"name": "heldout", "variants": ["canonical", "bare", "payload_only"], "loglik": true}, …] | null,
 "primary_set": "heldout", "primary_variant": "canonical",
 "headline": {"base_points": 38.0, "run_points": 41.0, "max_points": 70.0, "delta_points": 3.0, "delta_score_pct": 4.29,
              "base_parse_fail_rate": 0.05, "run_parse_fail_rate": 0.0, "delta_lenient_points": 1.0},
 "sets": {"<set>": {"variants": {"<variant>": {"base_points", "run_points", "max_points", "delta_points", "base_score_pct", "run_score_pct", "delta_score_pct",
                                              "base_parse_fail_rate", "run_parse_fail_rate", "delta_lenient_points"}},
                    "loglik": {"base_acc", "run_acc", "delta_acc_pct"} | null,
                    "organizer": {"base_accuracy", "run_accuracy", "delta_acc_pct", "base_mean_abc_mass", "run_mean_abc_mass",
                                  "delta_mean_abc_mass", "rotated_delta_acc_pct" | null}}},   # key ABSENT unless both sides measured it
 "per_subject": {"historia": {"base_points", "run_points", "max_points", "delta_points"}},        # primary set/variant
 "paired": {"wins": 5, "losses": 2, "ties": 53, "sign_test_p": 0.45, "bootstrap_ci95_delta_points": [-1.0, 6.0],
            "n_paired": 60, "n_unpaired": 0} | null,
 "general_heldout": {"base_nll": 1.20, "run_nll": 1.25, "rel_change": 0.0417} | null,
 "flipped_items": {"gained": ["his-03"], "lost": ["mat-02"]} | null}
```
`fingerprint_diff` = `{dotted.part: {"base", "run"}}` (empty on a match); `eval_limit` = the debug cap on items per set (null = full eval sets);
`eval_sets` = the sets/variants/loglik the run's eval was configured with (its fingerprint parts; null for old summaries);
`paired` / `flipped_items` are null when either side lacks `predictions.<primary_set>.<primary_variant>.jsonl`.

## Gate (`evaluate_gate`) — all must hold to promote

1. `headline.delta_points >= gate.min_delta_points` (base.yaml 2.0: one lucky flip on 60 items is not a gain)
2. `run_parse_fail_rate <= base_parse_fail_rate + gate.max_parse_fail_rate_increase` (primary set/variant)
3. `general_heldout.rel_change <= gate.max_general_nll_increase` (missing general NLL -> FAIL)
4. every variant of the primary set: `delta_points >= -gate.max_variant_regression_points`
5. every non-primary set, primary variant: `delta_score_pct >= -gate.max_secondary_score_drop_pct` (base.yaml 1.0 pp ~ 3.7 of 374 ext items)
5b. `headline.delta_score_pct + sum(min(0, secondary delta_score_pct)) >= gate.min_net_score_pct`: secondary drops must be paid for by primary gains
6. every set with loglik: `delta_acc_pct >= -gate.max_loglik_acc_drop_pct`
7. if `gate.must_beat_current_candidate`: `delta_points > candidate.delta_points` and `delta_points >= candidate.delta_points +
   gate.min_improvement_points`. Satisfied when `cmp.run_name == candidate.run_name` (the candidate itself re-recorded: `record_run` refuses a
   different model under the candidate's run name). FAILs when the candidate is not base and `candidate.fingerprint != cmp.fingerprint`
   (deltas measured under different evals are not comparable: re-evaluate the candidate first; the pipeline does this automatically).
8. `headline.delta_lenient_points >= -gate.max_lenient_regression_points`, and every non-primary set (primary variant):
   `100 * delta_lenient_points / max_points >= -gate.max_secondary_score_drop_pct` (strict stays the headline, but the organizers' grader may be
   lenient: a format gain must not hide lost knowledge)
9. every set with an `organizer` block in BOTH summaries: `organizer.delta_acc_pct >= -gate.max_organizer_acc_drop_pct`
   (3.0 pp). A set without organizer data on either side yields NO line at all (never a PASS, never a FAIL), so an eval
   run before the protocol existed still promotes. The threshold key is optional in a `gate` dict
   (`registry.DEFAULT_MAX_ORGANIZER_ACC_DROP_PCT` = 3.0): an old checkpoint's `resolved_config.yaml` keeps working.
- `[fp]` `fingerprint_match` must be true: a comparison made with `--allow-fingerprint-mismatch` is recorded but never auto-promoted.
- `[limit]` `eval_limit` must be null: a debug eval capped by `eval.limit` (e.g. the smoke config) is recorded but never auto-promoted.
- `[scope]` (`required_scope`, always applied by `record_run`): every set of configs/base.yaml `eval.sets` (optional ones only when their file
  exists) must be in `cmp.eval_sets` with all its variants and, when configured, loglik: a narrower comparison is recorded but never auto-promoted.
  An organizer-only set (spec with `variants: []` and no loglik, e.g. the prawko sets) requires nothing here — rule 9 judges it, and rule 9 is
  skippable, so a missing organizer eval never blocks a promotion.

Every check yields one `PASS [rule] …` / `FAIL [rule] …` line (stored as `gate_reasons`). `scripts/promote.py --force` can still promote a
failed run (logged as `forced`). `scripts/compare.py` never records limited or `-dbg` evals at all (it prints the gate as a dry run).
**Demotion**: when the current candidate itself is re-recorded and fails the gate on a comparable measurement (no `[fp]`/`[limit]`/`[scope]`
failure), `record_run` replaces it with the untouched base model (CANDIDATE.json reason `candidate <run> failed re-evaluation (rules …)`,
run row `demoted: true`); older passing runs are not reinstated (they were measured under a possibly stale eval).

## Registry files (`results/<base_slug>/`, written by `train/registry.py`)

Paths inside rows and CANDIDATE.json are repo-relative when inside the repo (results/ is shared between machines), else absolute.

**CANDIDATE.json**: `{"run_name", "model_path", "adapter_path"?, "delta_points", "score_pct", "promoted_at", "tags", "reason", "fingerprint",
"manual"?, "forced"?, "warnings"?}`. For the base model `run_name` is `base` and `model_path` is the hub id. `warnings` (manual and gate
promotions) flags e.g. a checkpoint without a merged model. `<checkpoints_root>/CANDIDATE-<base_slug>` is a relative symlink to that base
model's candidate checkpoint dir (absent for base); promoting or rolling back one base model never touches another base model's link.

**runs.jsonl** (append-only). `"event": "run"` rows (record_run):
`run_name, timestamp, base_model, model_path, adapter_path, run_dir, checkpoints_root, fingerprint, fingerprint_match, primary_set,
primary_variant, base_points, run_points, max_points, delta_points, score_pct, delta_score_pct, parse_fail_rate, base_parse_fail_rate,
delta_lenient_points, secondary_delta_score_pct {set: pp}, loglik_delta_acc_pct {set: pp}, organizer_delta_acc_pct {set: pp},
organizer_abc_mass {set: the run's mean probability mass on the option letters}, general_nll_rel_change, ci95, wins, losses, ties,
sign_test_p, gate_passed, gate_failed_rules, gate_reasons, promoted, demoted, previous_candidate, tags, notes, config_hash, config_path,
hparams {lr, r, alpha, epochs, general_ratio, general_ratio_trained, variant_mix, n_train}, train {n_train, dropped_too_long,
dropped_too_long_by_split, dropped_too_long_by_kind, loss_tokens, runtime_s, device, quantization, dtype, step0, last_epoch},
dedup (counts from data/synthetic/dedup_report.json)`.
`"event": "manual_promote"` rows (promote): `run_name, timestamp, reason, tags, force, forced, gate_passed, model_path, previous_candidate, warnings`.

**tags.json**: `{tag: run_name}` for tags of promoted candidates. **LEADERBOARD.md**: latest row per run + base, ★ on the candidate (regenerated);
its `org Δ%` column is `organizer_delta_acc_pct` per set (`—` for rows recorded without organizer data).

## File ownership while workflows run in parallel

- Contract files (read-only for builders; report needed changes instead of editing): `eval/{schema,answer_format,grader,prompts,modeling}.py`,
  `train/{config,wiki,synth_prompt}.py`, `scripts/{fetch_wiki,show_items}.py`, `configs/base.yaml`, `docs/ARCHITECTURE.md`, `data/eval/{heldout,dev,verification_log}.jsonl`.
- Data-generation workflow owns: `data/wiki/*`, `data/synthetic/raw/*`, `data/synthetic/verify/*`.
- Each builder owns only the files listed in its task.
