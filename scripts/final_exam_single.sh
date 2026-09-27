#!/usr/bin/env bash
# The exam with ONE model for everything (images included), for a submission that may declare a single model:
#   MODEL=gemma  (default) Gemma-4-12B-it IQ4_XS, multimodal: it describes the images itself (its own projector), then
#                answers. Base = the untouched model on the organizers' protocol (thinking off, no harness); ours =
#                thinking on + local RAG with relevance cutoff + label repair + essay retry.
#   MODEL=bielik Bielik-11B Q4_K_S, text only: no image descriptions at all (the images stay unseen markers).
#                Base = organizers' protocol; ours = the Bielik harness (RAG cutoff, 9 votes, label repair, planned essay).
#
#   bash scripts/final_exam_single.sh exams/final final            # Sunday (Gemma)
#   MODEL=bielik bash scripts/final_exam_single.sh exams/mock one-bielik
#
# Outputs runs/<tag>-base/answers.json and runs/<tag>-tuned/answers.json; one model server at a time, each checked
# against the 8.8 GB team cap (scripts/serve_model.sh --mem-check). Requests use cache_prompt false (reproducible).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXAM_DIR="${1:?usage: final_exam_single.sh <exam dir> <tag>}"
TAG="${2:?usage: final_exam_single.sh <exam dir> <tag>}"
MODEL="${MODEL:-gemma}"
MODELS="${MODELS_DIR:-/workspace/models}"
GEMMA_GGUF="${GEMMA_GGUF:-$MODELS/gemma-4-12b/gemma-4-12b-it-IQ4_XS.gguf}"
GEMMA_MMPROJ="${GEMMA_MMPROJ:-$MODELS/gemma-4-12b/mmproj-F16.gguf}"
BIELIK_GGUF="${BIELIK_GGUF:-$MODELS/bielik-11b/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf}"
if [[ -z "${PYTHON:-}" && -x /workspace/venv-exam/bin/python ]]; then PYTHON=/workspace/venv-exam/bin/python; fi
PY="${PYTHON:-python}"
T0=$(date +%s)
say() { echo "[$(date +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }
die() {  # stop every server this script may have started, so none lingers holding memory or a port
  say "FAILED: $*"
  for s in vlm bielik gemma gvlm gbase gours; do tmux kill-session -t "=$s" 2>/dev/null; done
  exit 1
}
stop_server() { tmux kill-session -t "=$1" 2>/dev/null; sleep 3; }
start_server() {  # $1 session, $2 port, rest: serve_model.sh args
  local s="$1" port="$2"; shift 2
  stop_server "$s"
  tmux new -d -s "$s" "bash scripts/serve_model.sh --port $port --parallel 1 $* 2>&1 | tee runs/$TAG-$s.log"
  for _ in $(seq 1 150); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && return 0; sleep 2; done
  die "server $s did not come up (runs/$TAG-$s.log)"
}
mem_ok() {  # the output is captured first: with pipefail, `| grep -q` would make the check never fire
  local out
  out="$(bash scripts/serve_model.sh --mem-check 2>&1)"
  echo "$out" >> "runs/$TAG-mem.log"
  if grep -q "OVER" <<<"$out"; then die "memory over the team cap: $(grep OVER <<<"$out" | head -1)"; fi
}

[[ -f "$EXAM_DIR/exam.json" ]] || die "$EXAM_DIR/exam.json not found"
EXAM_NAME="$(basename "$EXAM_DIR")"
say "exam $EXAM_DIR, tag $TAG, single model: $MODEL"

if [[ "$MODEL" == gemma ]]; then
  [[ -f "$GEMMA_GGUF" && -f "$GEMMA_MMPROJ" ]] || die "Gemma files missing"
  DESC="runs/desc/$EXAM_NAME-gemma.json"
  # 1. Gemma describes the images (vision projector on, thinking off)
  # 4k context per request is enough for one image + the instruction + the description, and keeps it under the cap
  # Gemma encodes an image as one non-causal block: the micro-batch must hold all its tokens (else llama.cpp aborts with
  # "non-causal attention requires n_ubatch >= n_tokens"), so cap image tokens at 1024 and use 1024-token batches
  start_server gvlm 8081 --vlm --model "$GEMMA_GGUF" --mmproj "$GEMMA_MMPROJ" --alias vlm --ctx 4096 -- \
    -ub 1024 -b 1024 --image-max-tokens 1024
  mem_ok
  $PY scripts/describe_images.py --exam-dir "$EXAM_DIR" --cache "$DESC" --parallel 1 \
    || $PY scripts/describe_images.py --exam-dir "$EXAM_DIR" --cache "$DESC" --parallel 1 || die "image descriptions"
  mem_ok
  stop_server gvlm
  say "descriptions: $DESC"
  # 2. untouched base: organizers' protocol, thinking off, no harness
  start_server gbase 8083 --model "$GEMMA_GGUF" --alias gemma -- --reasoning-budget 0 \
    --chat-template-kwargs '{"enable_thinking":false}'
  mem_ok
  $PY scripts/run_exam.py --exam-dir "$EXAM_DIR" --label "$TAG-base" --model gemma --base-url http://127.0.0.1:8083/v1 \
    --parallel 1 --descriptions "$DESC" || die "base run"
  stop_server gbase
  say "base done"
  # 3. ours: thinking on (12k context keeps it under the cap), RAG with cutoff, label repair, essay retry
  start_server gours 8083 --model "$GEMMA_GGUF" --alias gemma --ctx 12288
  mem_ok
  $PY scripts/run_exam.py --exam-dir "$EXAM_DIR" --label "$TAG-tuned" --model gemma --base-url http://127.0.0.1:8083/v1 \
    --parallel 1 --descriptions "$DESC" --style rag --rag-index data/rag/index --rag-min-score 50 --repair-labels \
    --essay-retry --max-tokens-short 4096 --max-tokens-essay 8192 || die "tuned run"
  mem_ok
  stop_server gours
elif [[ "$MODEL" == bielik ]]; then
  [[ -f "$BIELIK_GGUF" ]] || die "Bielik file missing"
  DESC="runs/desc/$EXAM_NAME-none.json"
  echo '{}' > "$DESC"          # no descriptions: the images stay as unseen [Obraz: ...] markers
  start_server bielik 8080 --model "$BIELIK_GGUF"
  mem_ok
  $PY scripts/run_exam.py --exam-dir "$EXAM_DIR" --label "$TAG-base" --parallel 1 --descriptions "$DESC" || die "base run"
  say "base done"
  $PY scripts/run_exam.py --exam-dir "$EXAM_DIR" --label "$TAG-tuned" --parallel 1 --descriptions "$DESC" --style rag \
    --rag-index data/rag/index --rag-min-score 50 --essay-mode plan --vote 9 --repair-labels --essay-retry || die "tuned run"
  stop_server bielik
else
  die "MODEL must be gemma or bielik"
fi

$PY scripts/validate_answers.py "runs/$TAG-base/answers.json" --exam-dir "$EXAM_DIR" || die "base file invalid"
$PY scripts/validate_answers.py "runs/$TAG-tuned/answers.json" --exam-dir "$EXAM_DIR" || die "tuned file invalid"
say "DONE in $(( $(date +%s) - T0 ))s"
echo "upload base : runs/$TAG-base/answers.json"
echo "upload tuned: runs/$TAG-tuned/answers.json"
if [[ "$MODEL" == gemma ]]; then echo "model       : google/gemma-4-12b-it (unsloth IQ4_XS GGUF + mmproj-F16), the only model"
else echo "model       : speakleash/Bielik-11B-v3.0-Instruct (bartowski Q4_K_S GGUF), the only model"; fi
