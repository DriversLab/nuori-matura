#!/usr/bin/env python
"""Merge a LoRA adapter into its unquantized base model and export an eval/submission-ready model dir.

  python scripts/merge_and_export.py --adapter checkpoints/run1/adapter --out checkpoints/run1
  python scripts/merge_and_export.py --adapter path/to/adapter --out export/run1 --base speakleash/Bielik-4.5B-v3.0-Instruct

--out must be the adapter's own checkpoint dir (rebuilding its merged weights) or a new dir; another run's checkpoint
dir is refused. An 11B bf16 merge on a <= 24 GB GPU offloads layers to CPU (slow); --device cpu needs ~2x the weights in RAM.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import traceback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (contains adapter_config.json)")
    ap.add_argument("--out", required=True, help="output dir for the merged model: the adapter's own checkpoint dir or a new dir")
    ap.add_argument("--base", default=None, help="base model id/path (default: base_model_name_or_path of the adapter)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="dtype of the merged weights")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="merge device (auto: cuda if available, else cpu)")
    args = ap.parse_args()

    from train.merge import merge_and_export

    out = merge_and_export(args.adapter, args.out, base_model=args.base, dtype=args.dtype, device=args.device)
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    print(f"merged model written to {out}")
    print("files: " + ", ".join(files))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(f"merge_and_export.py: FAILED: {type(exc).__name__}: {exc}")
