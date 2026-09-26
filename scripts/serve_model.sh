#!/usr/bin/env bash
# Serve Bielik-4.5B-v3.0-Instruct (official Q8_0 GGUF, optional GGUF LoRA) or the image-description VLM through an
# OpenAI-compatible API. It runs in the foreground: start it in tmux and stop it with Ctrl-C.
#
#   bash scripts/serve_model.sh                                      # base model, llama-server, 127.0.0.1:8080
#   bash scripts/serve_model.sh --lora checkpoints/history-v0/lora-f16.gguf
#        # adapter loaded at scale 0: a request is the base unless it sends "lora": [{"id": 0, "scale": 1}]
#   bash scripts/serve_model.sh --lora X.gguf --lora-apply           # adapter at scale 1 for every request
#   bash scripts/serve_model.sh --vlm                                # image-description VLM on 127.0.0.1:8081
#   bash scripts/serve_model.sh --engine qvac [--lora X.gguf]        # QVAC: qvac serve --openai (writes a qvac.config.json)
#   bash scripts/serve_model.sh --dry-run ...                        # print the command and the memory estimate only
#   bash scripts/serve_model.sh --mem-check                          # memory of the running servers against the
#                                                                    # 8.8 GB team cap (8 GB + 10%, host RSS + GPU)
#
# Options
#   --engine llama|qvac   llama = llama-server (upstream llama.cpp, or the QVAC Fabric build via --llama-bin; default)
#                         qvac  = `qvac serve --openai` (npm @qvac/cli). It ignores logprobs/stop/n>1, defaults to
#                                 temp 0.8 and repeat_penalty 1.1, and has no per-request LoRA: send sampling params in
#                                 every request and restart the server to switch between base and tuned
#   --model PATH          GGUF (default $MODELS_DIR/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf;
#                         --vlm: $MODELS_DIR/vlm/qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf, the pick in docs/VISION.md)
#   --mmproj PATH         VLM projector (default $MODELS_DIR/vlm/qwen3.5-4b/mmproj-F16.gguf)
#   --lora PATH           GGUF LoRA adapter (scripts/export_lora_gguf.sh). llama: global scale 0 unless --lora-apply
#   --lora-apply          llama: load the adapter at scale 1 (requests without a "lora" field get the tuned model)
#   --ctx N               context PER SLOT (default 8192 = organizers' protocol); the server gets -c N*parallel
#   --parallel N          slots / concurrent requests (default 4 text, 2 vlm; 1 = most reproducible, least RAM)
#   --backend B           auto|cuda|metal|vulkan|cpu (default auto)
#   --kv K[,V]            KV-cache types (default q8_0 on cuda/metal; tbq4_0,pq4_0 on vulkan/cpu when the binary
#                         supports TurboQuant, i.e. the QVAC Fabric build or qvac; else q8_0)
#   --gpu-layers N        default 99 (0 on cpu)
#   --host H / --port P   default 127.0.0.1 / 8080 (vlm 8081)
#   --llama-bin DIR       dir with llama-server (default $LLAMA_CPP_DIR/build/bin, else PATH)
#   --alias NAME          model name the API reports (default bielik, vlm: vlm = describe_images.py's default)
#   -- ARGS...            passed through to llama-server unchanged (e.g. -- --image-max-tokens 1120 for Gemma)
#
# --vlm on llama-server also turns thinking off (--reasoning-budget 0, enable_thinking=false) and, for Qwen models,
# sets --image-min-tokens 1024 --image-max-tokens 2048 (docs/VISION.md, "Launch").
#
# Env: MODELS_DIR (default /workspace/models on Forgehand, else ~/models), LLAMA_CPP_DIR (default $WORK/llama.cpp)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /workspace ]]; then WORK=/workspace; else WORK="$HOME"; fi
MODELS_DIR="${MODELS_DIR:-$WORK/models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORK/llama.cpp}"
BASE_GGUF="Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"
BASE_GGUF_SHA256="562f2291de257890adf2b4a914da8b194affe6a7a838a6b7ef3d342f306c1b7f"  # HF LFS oid of the official file
VLM_GGUF="vlm/qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf"          # unsloth/Qwen3.5-4B-GGUF (apache-2.0), 4.48 GB
VLM_MMPROJ="vlm/qwen3.5-4b/mmproj-F16.gguf"              # 0.67 GB; keep the projector at F16 (OCR quality)
RAM_CAP_GB="${RAM_CAP_GB:-8.8}" # team cap on the running model: 8 GB + 10%, decimal GB (host RSS + GPU)

ENGINE=llama MODE=text MODEL="" MMPROJ="" LORA="" LORA_APPLY=0 CTX=8192 PARALLEL="" BACKEND=auto KV=""
GPU_LAYERS="" HOST=127.0.0.1 PORT="" LLAMA_BIN="" ALIAS="" DRY_RUN=0 EXTRA=()

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "ERROR: $*" >&2; exit 1; }

mem_check() {
  echo "== memory of running model servers (team cap: ${RAM_CAP_GB} GB) =="
  local pids
  pids="$(pgrep -f 'llama-server|qvac serve|bare-runtime' || true)"
  if [[ -z "$pids" ]]; then echo "no llama-server / qvac process running"; fi
  local gpu="" over=0 rss_kb gpu_mib total
  command -v nvidia-smi >/dev/null 2>&1 && \
    gpu="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  for pid in $pids; do
    rss_kb="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')"
    [[ -n "$rss_kb" ]] || continue
    gpu_mib="$(awk -F', *' -v p="$pid" '$1 == p { s += $2 } END { print s + 0 }' <<<"$gpu")"
    # strict: decimal GB (1e9 bytes), host RSS + the process's whole GPU allocation (CUDA context included)
    total="$(awk -v r="$rss_kb" -v g="$gpu_mib" 'BEGIN { printf "%.2f", (r * 1024 + g * 1048576) / 1e9 }')"
    printf "pid %s  host RSS %.2f GB + GPU %.2f GB = %s GB  " "$pid" \
      "$(awk -v r="$rss_kb" 'BEGIN { print r * 1024 / 1e9 }')" "$(awk -v g="$gpu_mib" 'BEGIN { print g * 1048576 / 1e9 }')" "$total"
    if awk -v t="$total" -v c="$RAM_CAP_GB" 'BEGIN { exit !(t > c) }'; then echo "OVER ${RAM_CAP_GB} GB"; over=1
    else echo "ok"; fi
    ps -o command= -p "$pid" 2>/dev/null | cut -c1-150 | sed 's/^/    /'
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "-- macOS: Metal buffers are in unified memory; also check Activity Monitor > Memory for llama-server --"
  fi
  echo "Count: host RSS + GPU memory of each model server, in decimal GB (1 GB = 1e9 bytes; nvidia-smi prints MiB)."
  echo "The VLM runs in a separate pre-pass and is stopped before the text model starts: the peak is the larger one."
  return "$over"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --vlm) MODE=vlm; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --mmproj) MMPROJ="$2"; shift 2 ;;
    --lora) LORA="$2"; shift 2 ;;
    --lora-apply) LORA_APPLY=1; shift ;;
    --ctx) CTX="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --kv) KV="$2"; shift 2 ;;
    --gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --llama-bin) LLAMA_BIN="$2"; shift 2 ;;
    --alias) ALIAS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --mem-check) mem_check; exit $? ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

[[ "$ENGINE" == llama || "$ENGINE" == qvac ]] || die "--engine must be llama or qvac"
if [[ "$MODE" == vlm ]]; then
  MODEL="${MODEL:-$MODELS_DIR/$VLM_GGUF}"
  MMPROJ="${MMPROJ:-$MODELS_DIR/$VLM_MMPROJ}"
  PORT="${PORT:-8081}"; PARALLEL="${PARALLEL:-2}"; ALIAS="${ALIAS:-vlm}"
  [[ -z "$LORA" ]] || die "--lora is for the text model, not --vlm"
else
  MODEL="${MODEL:-$MODELS_DIR/$BASE_GGUF}"
  PORT="${PORT:-8080}"; PARALLEL="${PARALLEL:-4}"; ALIAS="${ALIAS:-bielik}"
fi
[[ -f "$MODEL" ]] || die "model not found: $MODEL (run scripts/setup_labqoat.sh or pass --model)"
[[ -z "$MMPROJ" || -f "$MMPROJ" ]] || die "mmproj not found: $MMPROJ"
[[ -z "$LORA" || -f "$LORA" ]] || die "LoRA adapter not found: $LORA (bash scripts/export_lora_gguf.sh <adapter dir>)"
[[ "$CTX" =~ ^[0-9]+$ && "$PARALLEL" =~ ^[0-9]+$ && "$PARALLEL" -ge 1 ]] || die "--ctx and --parallel must be positive integers"
TOTAL_CTX=$((CTX * PARALLEL))

# ---------------------------------------------------------------- backend + KV cache
if [[ "$BACKEND" == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then BACKEND=cuda
  elif [[ "$(uname -s)" == "Darwin" ]]; then BACKEND=metal
  elif command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then BACKEND=vulkan
  else BACKEND=cpu
  fi
fi
case "$BACKEND" in cuda|metal|vulkan|cpu) ;; *) die "--backend must be auto|cuda|metal|vulkan|cpu" ;; esac
if [[ -z "$GPU_LAYERS" ]]; then
  if [[ "$BACKEND" == cpu ]]; then GPU_LAYERS=0; else GPU_LAYERS=99; fi
fi

SERVER_HELP=""
if [[ "$ENGINE" == llama ]]; then
  if [[ -n "$LLAMA_BIN" ]]; then SERVER="$LLAMA_BIN/llama-server"
  elif [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]]; then SERVER="$LLAMA_CPP_DIR/build/bin/llama-server"
  else SERVER="$(command -v llama-server || true)"
  fi
  [[ -n "$SERVER" && -x "$SERVER" ]] || die "llama-server not found (build it with scripts/setup_labqoat.sh or pass --llama-bin)"
  SERVER_HELP="$("$SERVER" --help 2>&1 || true)"
fi
supports() { grep -q -- "$1" <<<"$SERVER_HELP"; }

if [[ -z "$KV" ]]; then
  if [[ "$BACKEND" == cuda || "$BACKEND" == metal ]]; then KV="q8_0,q8_0"
  elif [[ "$ENGINE" == qvac ]] || supports "tbq4_0"; then KV="tbq4_0,pq4_0"
  else KV="q8_0,q8_0"
  fi
fi
KV_K="${KV%%,*}"; KV_V="${KV#*,}"; [[ "$KV" == *,* ]] || KV_V="$KV_K"
if [[ "$KV_K$KV_V" == *bq* || "$KV_K$KV_V" == *pq* ]]; then
  [[ "$BACKEND" == vulkan || "$BACKEND" == cpu ]] || die "TurboQuant KV ($KV) has no CUDA/Metal kernels: use --kv q8_0 on $BACKEND"
  if [[ "$ENGINE" == llama ]] && ! supports "tbq4_0"; then
    die "$SERVER does not know $KV: use the QVAC Fabric build (--llama-bin \$WORK/qvac-fabric-llm.cpp/build/bin) or --kv q8_0"
  fi
fi

# ---------------------------------------------------------------- memory estimate (weights + KV; compute buffers extra)
PY="python3"; [[ -x "$REPO_ROOT/.venv/bin/python" ]] && PY="$REPO_ROOT/.venv/bin/python"
file_bytes() { if [[ -n "$1" && -f "$1" ]]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }
W_BYTES=$(( $(file_bytes "$MODEL") + $(file_bytes "$MMPROJ") + $(file_bytes "$LORA") ))
KV_BYTES="$("$PY" "$REPO_ROOT/scripts/check_tokenizer.py" --gguf-meta "$MODEL" --kv-ctx "$TOTAL_CTX" --kv-types "$KV_K,$KV_V" 2>/dev/null \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["kv_bytes"])' 2>/dev/null || echo "")"
if [[ -z "$KV_BYTES" ]]; then KV_BYTES=0; KV_NOTE="(KV estimate unavailable)"; else KV_NOTE=""; fi
EST=$(awk -v w="$W_BYTES" -v k="$KV_BYTES" 'BEGIN { printf "%.2f", (w + k) / 1e9 + 0.5 }')

echo "== serve_model: $MODE via $ENGINE on $BACKEND =="
echo "model:     $MODEL"
[[ "$MODEL" == *"$BASE_GGUF" ]] && echo "           official file sha256 should be $BASE_GGUF_SHA256 (organizers check the untouched weights)"
[[ -n "$MMPROJ" ]] && echo "mmproj:    $MMPROJ"
if [[ -n "$LORA" ]]; then
  if [[ "$ENGINE" == qvac || "$LORA_APPLY" == 1 ]]; then echo "lora:      $LORA (applied, scale 1, to every request)"
  else echo "lora:      $LORA (loaded at scale 0; tuned requests must send \"lora\": [{\"id\": 0, \"scale\": 1}])"; fi
fi
echo "context:   $CTX per slot x $PARALLEL slots = $TOTAL_CTX tokens; KV cache $KV_K/$KV_V"
echo "endpoint:  http://$HOST:$PORT/v1  (model name: $ALIAS)"
echo "memory:    ~$EST GB = weights $(awk -v b="$W_BYTES" 'BEGIN { printf "%.2f", b / 1e9 }') GB + KV $(awk -v b="$KV_BYTES" 'BEGIN { printf "%.2f", b / 1e9 }') GB + ~0.5 GB compute buffers (estimate) $KV_NOTE"
if awk -v e="$EST" -v c="$RAM_CAP_GB" 'BEGIN { exit !(e > c) }'; then
  echo "WARNING:   estimate above the ${RAM_CAP_GB} GB cap: lower --parallel, or use --kv q4_0 (cuda/metal) / tbq4_0,pq4_0 (vulkan/cpu)"
fi
echo "check:     bash scripts/serve_model.sh --mem-check   (host RSS + nvidia-smi per process, once the model is loaded)"
echo "requests:  send temperature 0, seed 42, repeat_penalty 1.0 and max_tokens (2048; 4096 for the essay) explicitly"

# ---------------------------------------------------------------- llama-server
if [[ "$ENGINE" == llama ]]; then
  ARGS=(-m "$MODEL" --host "$HOST" --port "$PORT" --alias "$ALIAS"
        -c "$TOTAL_CTX" --parallel "$PARALLEL" -ngl "$GPU_LAYERS"
        --temp 0 --seed 42 --repeat-penalty 1.0
        -ctk "$KV_K" -ctv "$KV_V")
  if supports "flash-attn \[on"; then ARGS+=(-fa on); else ARGS+=(-fa); fi   # a quantized V cache needs flash attention
  supports "--jinja" && ARGS+=(--jinja)                                       # the GGUF's own chat template
  supports "--no-kv-unified" && ARGS+=(--no-kv-unified)                       # each slot owns exactly CTX tokens
  supports "--cache-ram" && ARGS+=(--cache-ram 0)                             # no host-RAM prompt cache (RAM cap)
  if [[ "$MODE" == vlm ]]; then
    ARGS+=(--mmproj "$MMPROJ")
    supports "--reasoning-budget" && ARGS+=(--reasoning-budget 0)
    supports "--chat-template-kwargs" && ARGS+=(--chat-template-kwargs '{"enable_thinking":false}')
    shopt -s nocasematch
    if [[ "$(basename "$MODEL")" == qwen* ]] && supports "--image-min-tokens"; then
      ARGS+=(--image-min-tokens 1024 --image-max-tokens 2048)
    fi
    shopt -u nocasematch
  fi
  if [[ -n "$LORA" ]]; then
    ARGS+=(--lora "$LORA")
    [[ "$LORA_APPLY" == 1 ]] || ARGS+=(--lora-init-without-apply)
  fi
  [[ ${#EXTRA[@]} -eq 0 ]] || ARGS+=("${EXTRA[@]}")
  echo "command:   $SERVER ${ARGS[*]}"
  echo "verify:    curl -s http://$HOST:$PORT/health; curl -s http://$HOST:$PORT/lora-adapters"
  [[ "$DRY_RUN" == 1 ]] && exit 0
  exec "$SERVER" "${ARGS[@]}"
fi

# ---------------------------------------------------------------- qvac serve --openai
command -v qvac >/dev/null 2>&1 || [[ "$DRY_RUN" == 1 ]] || die "qvac not found: npm i -g @qvac/cli (Node.js >= 18)"
STATE_DIR="${SERVE_STATE_DIR:-$WORK/.serve}"
mkdir -p "$STATE_DIR"
CONFIG="$STATE_DIR/qvac.config.$MODE.json"
DEVICE=gpu; [[ "$BACKEND" == cpu ]] && DEVICE=cpu
"$PY" - "$CONFIG" "$ALIAS" "$MODEL" "$MMPROJ" "$LORA" "$DEVICE" "$GPU_LAYERS" "$TOTAL_CTX" "$PARALLEL" "$KV_K" "$KV_V" <<'PYEOF'
import json, sys
out, alias, model, mmproj, lora, device, ngl, ctx, parallel, kv_k, kv_v = sys.argv[1:]
# Keys follow the @qvac/llm-llamacpp config table (values as strings, as that README requires). If `qvac serve`
# rejects a key, run `qvac configure` to see the accepted modelConfig schema and adjust this block.
config = {"device": device, "gpu_layers": ngl, "ctx_size": ctx, "parallel": parallel,
          "temp": "0", "top_k": "1", "seed": "42", "repeat_penalty": "1.0", "flash-attn": "on",
          "cache-type-k": kv_k, "cache-type-v": kv_v}
if lora:
    config["lora"] = lora
if mmproj:
    config["projectionModelSrc"] = mmproj
    config["reasoning_budget"] = "0"  # Qwen3.5 thinks by default (docs/VISION.md)
entry = {"src": model, "type": "llm", "preload": True, "default": True, "config": config}
json.dump({"serve": {"models": {alias: entry}}}, open(out, "w"), indent=2)
print(f"qvac config: {out}")
PYEOF
echo "command:   qvac serve --openai --config $CONFIG --host $HOST --port $PORT --model $ALIAS"
[[ "$DRY_RUN" == 1 ]] && exit 0
exec qvac serve --openai --config "$CONFIG" --host "$HOST" --port "$PORT" --model "$ALIAS"
