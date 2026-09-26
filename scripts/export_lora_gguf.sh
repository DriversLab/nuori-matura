#!/usr/bin/env bash
# Convert a PEFT LoRA adapter dir into a GGUF LoRA adapter with llama.cpp's convert_lora_to_gguf.py.
#
#   bash scripts/export_lora_gguf.sh checkpoints/history-v0/adapter
#   bash scripts/export_lora_gguf.sh checkpoints/history-v0/adapter /workspace/models/history-v0-lora-f16.gguf
#
# Output (default): <adapter dir>/../lora-f16.gguf, e.g. checkpoints/history-v0/lora-f16.gguf (checkpoints/ is gitignored).
# Serve it with: bash scripts/serve_model.sh --lora <that file>
#
# Environment (all optional):
#   LLAMA_CPP_DIR   llama.cpp checkout that holds convert_lora_to_gguf.py (default: $WORK/llama.cpp, see below)
#   HF_BASE         base model: a local HF dir with config.json (preferred) or a hub id
#                   (default: $MODELS_DIR/Bielik-4.5B-v3.0-Instruct if it exists, else speakleash/Bielik-4.5B-v3.0-Instruct,
#                   which is gated and needs `hf auth login`)
#   OUTTYPE         f16 (default) | bf16 | f32 | q8_0
#   CONVERT_PYTHON  python to run the converter (default: $LLAMA_CPP_DIR/.venv-convert/bin/python made by
#                   setup_labqoat.sh, else the repo .venv, else python3). llama.cpp pins torch/transformers versions
#                   of its own, so the converter gets a separate venv instead of downgrading the training venv.
#
# Size note: an r=16 all-linear adapter for the 4.5B model is ~50M params, ~100 MB as f16. The 8 GB on disk
# (organizers' limit) is about the base weights (the registered model: official Q8_0 GGUF, 5.06 GB); base + adapter
# stays ~5.2 GB either way. The runtime check against the 8.8 GB team cap (8 GB + 10%, host RSS + GPU) is done on the
# running server (serve_model.sh --mem-check).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /workspace ]]; then WORK=/workspace; else WORK="$HOME"; fi
MODELS_DIR="${MODELS_DIR:-$WORK/models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORK/llama.cpp}"
OUTTYPE="${OUTTYPE:-f16}"
HF_BASE_ID="speakleash/Bielik-4.5B-v3.0-Instruct"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -lt 1 ]]; then usage >&2; exit 1; fi
ADAPTER_DIR="${1%/}"
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "ERROR: $ADAPTER_DIR/adapter_config.json not found (expected a PEFT adapter dir, e.g. checkpoints/<run>/adapter)" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" && ! -f "$ADAPTER_DIR/adapter_model.bin" ]]; then
  echo "ERROR: no adapter_model.safetensors / adapter_model.bin in $ADAPTER_DIR" >&2
  exit 1
fi
OUT="${2:-$(dirname "$ADAPTER_DIR")/lora-${OUTTYPE}.gguf}"

CONVERTER="$LLAMA_CPP_DIR/convert_lora_to_gguf.py"
if [[ ! -f "$CONVERTER" ]]; then
  echo "ERROR: $CONVERTER not found. Set LLAMA_CPP_DIR or run scripts/setup_labqoat.sh (clones llama.cpp)." >&2
  exit 1
fi

if [[ -n "${CONVERT_PYTHON:-}" ]]; then
  PY="$CONVERT_PYTHON"
elif [[ -x "$LLAMA_CPP_DIR/.venv-convert/bin/python" ]]; then
  PY="$LLAMA_CPP_DIR/.venv-convert/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="python3"
fi

HF_BASE="${HF_BASE:-}"
if [[ -z "$HF_BASE" ]]; then
  if [[ -f "$MODELS_DIR/Bielik-4.5B-v3.0-Instruct/config.json" ]]; then
    HF_BASE="$MODELS_DIR/Bielik-4.5B-v3.0-Instruct"
  else
    HF_BASE="$HF_BASE_ID"
  fi
fi
if [[ -d "$HF_BASE" ]]; then
  BASE_ARGS=(--base "$HF_BASE")
else
  BASE_ARGS=(--base-model-id "$HF_BASE")
fi

# The adapter must belong to the same base: compare adapter_config.json base_model_name_or_path with HF_BASE.
ADAPTER_BASE="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("base_model_name_or_path",""))' "$ADAPTER_DIR/adapter_config.json" 2>/dev/null || true)"
echo "adapter:        $ADAPTER_DIR (base_model_name_or_path: ${ADAPTER_BASE:-?})"
echo "base config:    ${BASE_ARGS[*]}"
echo "converter:      $CONVERTER (python: $PY)"
echo "outtype/outfile: $OUTTYPE -> $OUT"
case "$ADAPTER_BASE" in
  *Bielik-4.5B-v3.0-Instruct*|"") ;;
  *) echo "WARNING: adapter was trained on '$ADAPTER_BASE', not Bielik-4.5B-v3.0-Instruct: serve it on the matching GGUF" >&2 ;;
esac

mkdir -p "$(dirname "$OUT")"
"$PY" "$CONVERTER" --outtype "$OUTTYPE" --outfile "$OUT" "${BASE_ARGS[@]}" "$ADAPTER_DIR"

if [[ ! -s "$OUT" ]]; then
  echo "ERROR: converter finished but $OUT is missing or empty" >&2
  exit 1
fi
BYTES=$(wc -c < "$OUT" | tr -d ' ')
SHA=$( (sha256sum "$OUT" 2>/dev/null || shasum -a 256 "$OUT") | cut -d' ' -f1)
echo
echo "GGUF LoRA adapter: $OUT"
echo "size:   $BYTES bytes ($(awk -v b="$BYTES" 'BEGIN { printf "%.1f MB", b / 1e6 }'))"
echo "sha256: $SHA"
echo "The adapter is loaded next to the unchanged base GGUF (Q8_0, 5.06 GB); its ~0.1 GB is not part of the base-weights"
echo "size and the running server is measured separately (serve_model.sh prints the memory check)."
echo "next: bash scripts/serve_model.sh --lora $OUT"
