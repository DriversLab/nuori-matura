#!/usr/bin/env python
"""Describe every exam image with a local VLM (pre-pass before Bielik runs). See docs/VISION.md.

Start the VLM server ALONE first (llama-server --mmproj ... --port 8081, or qvac serve --openai), run this, then
stop the VLM before loading Bielik.

  python scripts/describe_images.py --exam-dir exams/mock                  # cache: runs/desc/mock.json
  python scripts/describe_images.py --exam-dir exams/mock --dry-run        # no server: list images + first prompt
  python scripts/describe_images.py --exam-dir exams/mock --only images/Z01.png --force   # redo one image
  python scripts/describe_images.py --exam-dir exams/final --base-url http://127.0.0.1:11434 --model vlm  # QVAC

Then: python scripts/run_exam.py --exam-dir exams/mock --descriptions runs/desc/mock.json ...

Exit code: 0 all images described or cached, 1 some failed (rerun retries only those), 2 server unreachable.
The descriptions are derived from exam content: keep the cache out of git.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import time

from harness import vision


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam-dir", required=True, help="folder with exam.json and images/")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", vision.DEFAULT_BASE_URL),
                    help="OpenAI-compatible VLM server (default: $VLM_BASE_URL or %(default)s)")
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", vision.DEFAULT_MODEL),
                    help="model name/alias sent in requests; QVAC needs its serve.models alias (default: %(default)s)")
    ap.add_argument("--cache", help="descriptions cache JSON (default: runs/desc/<exam-dir name>.json; runs/ is gitignored)")
    ap.add_argument("--parallel", type=int, default=vision.DEFAULT_PARALLEL,
                    help="concurrent requests; match llama-server -np (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="ignore cached descriptions and describe again")
    ap.add_argument("--only", action="append", default=[], metavar="images/X.png", help="only this image (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=vision.DEFAULT_MAX_TOKENS, help="(default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=vision.DEFAULT_TIMEOUT_S, help="seconds per request (default: %(default)s)")
    ap.add_argument("--ocr", choices=("auto", "on", "off"), default="auto",
                    help="tesseract -l pol cross-check: auto = when installed (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="no requests: list images, cache hits and the first prompt")
    ap.add_argument("--reference", metavar="FILE",
                    help="JSON {path: text} (or a cache file) with reference descriptions; prints label/word recall "
                         "per image, e.g. to compare VLMs on the mock. Works on cached results without a server.")
    return ap.parse_args(argv)


def _load_reference(path: str) -> dict[str, str]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for k, v in data.items():
        text = v.get("description") if isinstance(v, dict) else v
        if isinstance(text, str) and text.strip():
            out[vision.normalize_image_path(k)] = text
    return out


def _print_reference_scores(results: list, reference: dict[str, str]) -> None:
    rows = []
    for r in results:
        ref = next((reference[p] for p in r.ref.paths if p in reference), None)
        if ref is None or not r.description:
            continue
        sc = vision.reference_scores(r.description, ref)
        rows.append((r.path, sc, len(r.description.split()), len(ref.split())))
    if not rows:
        print("reference: no overlapping described images")
        return
    print("\nreference comparison (label_recall = quoted labels found, word_recall = content-word stems found):")
    for path, sc, n_ours, n_ref in rows:
        lab = "  -  " if sc["label_recall"] is None else f"{sc['label_recall']:.2f}"
        print(f"  {path:<22} labels {lab}  words {sc['word_recall']:.2f}  ({n_ours} vs {n_ref} words)")
    labs = [sc["label_recall"] for _, sc, _, _ in rows if sc["label_recall"] is not None]
    words = [sc["word_recall"] for _, sc, _, _ in rows if sc["word_recall"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"  MEAN over {len(rows)} images: labels {mean(labs):.3f}  words {mean(words):.3f}")


def _fmt_s(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m{s:02d}s" if m else f"{seconds:.1f}s"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exam_dir = pathlib.Path(args.exam_dir)
    exam_file = exam_dir / "exam.json"
    if not exam_file.is_file():
        print(f"error: {exam_file} not found", file=sys.stderr)
        return 2
    exam = json.loads(exam_file.read_text(encoding="utf-8"))
    items = exam.get("items") or []
    cache_path = pathlib.Path(args.cache) if args.cache else ROOT / "runs" / "desc" / f"{exam_dir.resolve().name}.json"

    missing: list[str] = []
    refs = vision.collect_image_refs(exam_dir, items, missing=missing)
    for rel in missing:
        print(f"warning: {rel} is referenced but missing on disk; its marker stays as-is", file=sys.stderr)
    for ref in refs:
        if ref.declared_sha256 and ref.declared_sha256 != ref.sha256:
            print(f"warning: {ref.path} sha256 differs from exam.json (corrupted download?)", file=sys.stderr)
    if args.only:
        wanted = {vision.normalize_image_path(p) for p in args.only}
        refs = [r for r in refs if wanted & set(r.paths)]
    cache = vision.load_cache(cache_path)
    pending = [r for r in refs if args.force or vision.find_cached_entry(cache, r.sha256) is None]
    print(f"{exam.get('exam_id', '?')}: {len(refs)} unique images, {len(refs) - len(pending)} cached, "
          f"{len(pending)} to describe -> {cache_path}")

    if pending and not args.dry_run:
        problem = vision.check_server(args.base_url, api_key=os.environ.get(vision.API_KEY_ENV) or None)
        if problem:
            print(f"error: VLM server not reachable at {args.base_url}: {problem}\n"
                  "  start it first, e.g. llama-server -m <vlm.gguf> --mmproj <mmproj.gguf> -c 16384 -np 2 --port 8081"
                  " (see docs/VISION.md)", file=sys.stderr)
            return 2

    total = len(refs)
    done = 0

    def on_result(res: vision.ImageResult) -> None:
        nonlocal done
        done += 1
        items_s = ",".join(res.ref.item_ids)
        aliases = f" (= {', '.join(res.ref.paths[1:])})" if len(res.ref.paths) > 1 else ""
        line = f"[{done:>2}/{total}] {res.path:<22} {res.status:<9}"
        if res.status == "described":
            line += f" {_fmt_s(res.seconds):>7}  {len(res.description):>5} chars"
            cov = vision.ocr_coverage(res.description, res.ocr)
            if cov is not None:
                line += f"  ocr-coverage {cov:.2f}"
            if res.retried:
                line += "  [retried: looping output]"
            if res.truncated:
                line += "  [hit max_tokens]"
        elif res.status == "cached":
            line += f"          {len(res.description):>5} chars"
        line += f"  items {items_s}{aliases}"
        if res.error:
            line += f"  !! {res.error}"
        print(line, flush=True)

    t0 = time.monotonic()
    _, results = vision.describe_images_detailed(
        exam_dir, items, base_url=args.base_url, model=args.model, cache_path=cache_path, force=args.force,
        parallel=args.parallel, max_tokens=args.max_tokens, timeout=args.timeout, ocr=args.ocr,
        only=args.only or None, dry_run=args.dry_run, on_result=on_result)
    elapsed = time.monotonic() - t0

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "nothing to do"
    print(f"Done: {summary} in {_fmt_s(elapsed)} -> {cache_path}")

    if args.reference:
        _print_reference_scores(results, _load_reference(args.reference))

    if args.dry_run:
        planned = [r for r in results if r.status == "planned"]
        if planned:
            print(f"\n--- prompt for {planned[0].path} (the image goes first as a data URI) ---")
            print(vision.build_vlm_prompt(planned[0].ref))
        return 0
    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"{len(failed)} image(s) failed; rerun to retry only them: "
              + " ".join(f"--only {r.path}" for r in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
