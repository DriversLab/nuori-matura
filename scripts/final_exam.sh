#!/usr/bin/env bash
# The whole exam on the GPU box, one model server at a time (peak memory = one model, each checked against the cap):
#   1. images -> descriptions (Qwen3.5-4B VLM), then the VLM is stopped
#   2. untouched base: Bielik-11B Q4_K_S, organizers' protocol, server WITHOUT any adapter        -> runs/<tag>-base
#   3. our system, short items: Bielik-11B + harness (RAG with relevance cutoff, votes, label repair)
#      [+ LoRA if EXAM_LORA is set]                                                               -> runs/<tag>-items
#   4. [ESSAY_MODEL=gemma only] essay by Gemma-4-12B IQ4_XS with thinking                         -> runs/<tag>-essay
#   5. runs/<tag>-tuned/answers.json (= 3, or 3 merged with 4); validate base and tuned
#
#   bash scripts/final_exam.sh exams/final final            # Sunday
#   bash scripts/final_exam.sh exams/mock rehearsal          # timed rehearsal on the mock
#   EXAM_LORA=checkpoints/history-11b-v1/lora-f16.gguf bash scripts/final_exam.sh exams/final final
#
# Env: BIELIK_GGUF, GEMMA_GGUF (defaults below), EXAM_LORA (empty = no adapter), ESSAY_MODEL=bielik|gemma (default
# bielik: Bielik answers everything incl. the essay, one answering model; gemma: Gemma-4-12B writes the essay, only if
# the organizers confirm several models are fine), SKIP_VLM=1 (descriptions already in
# runs/desc/<exam folder>.json). Everything is logged to runs/<tag>.log; the upload files are printed at the end.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXAM_DIR="${1:?usage: final_exam.sh <exam dir> <tag>}"
TAG="${2:?usage: final_exam.sh <exam dir> <tag>}"
MODELS="${MODELS_DIR:-/workspace/models}"
BIELIK_GGUF="${BIELIK_GGUF:-$MODELS/bielik-11b/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf}"
GEMMA_GGUF="${GEMMA_GGUF:-$MODELS/gemma-4-12b/gemma-4-12b-it-IQ4_XS.gguf}"
EXAM_LORA="${EXAM_LORA:-}"
ESSAY_MODEL="${ESSAY_MODEL:-bielik}"   # bielik: one answering model (mock 46/60); gemma: essay by Gemma (mock 46/60)
[[ "$ESSAY_MODEL" == gemma || "$ESSAY_MODEL" == bielik ]] || { echo "ESSAY_MODEL must be gemma or bielik"; exit 1; }
DESC="runs/desc/$(basename "$EXAM_DIR").json"
# Python for the harness: the persistent exam venv (numpy + requests; survives a session restart, unlike /scratch)
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
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && return 0; sleep 2; done
  die "server $s did not come up (runs/$TAG-$s.log)"
}
mem_ok() {  # the output is captured first: with pipefail, `| grep -q` would make the check never fire
  local out
  out="$(bash scripts/serve_model.sh --mem-check 2>&1)"
  echo "$out" >> "runs/$TAG-mem.log"
  if grep -q "OVER" <<<"$out"; then die "memory over the team cap: $(grep OVER <<<"$out" | head -1)"; fi
}

[[ -f "$EXAM_DIR/exam.json" ]] || die "$EXAM_DIR/exam.json not found (unzip the pack into $EXAM_DIR/)"
[[ -f "$BIELIK_GGUF" && -f "$GEMMA_GGUF" ]] || die "model files missing"
[[ -z "$EXAM_LORA" || -f "$EXAM_LORA" ]] || die "EXAM_LORA=$EXAM_LORA not found"
ESSAY_IDS="$($PY -c "
import sys; sys.path.insert(0, '.')
from harness.exam_io import load_exam
from harness.prompts import item_kind
_, items = load_exam('$EXAM_DIR')
print(','.join(str(i['id']) for i in items if item_kind(i) == 'essay'))")"
say "exam $EXAM_DIR, tag $TAG, essay item(s): ${ESSAY_IDS:-none} by $ESSAY_MODEL, adapter: ${EXAM_LORA:-none}"

# 1. descriptions
if [[ "${SKIP_VLM:-0}" != 1 ]]; then
  start_server vlm 8081 --vlm
  mem_ok
  $PY scripts/describe_images.py --exam-dir "$EXAM_DIR" || $PY scripts/describe_images.py --exam-dir "$EXAM_DIR" \
    || die "image descriptions"
  stop_server vlm
fi
[[ -f "$DESC" ]] || die "$DESC missing"
say "descriptions: $DESC"

# 2. untouched base (no adapter loaded at all)
start_server bielik 8080 --model "$BIELIK_GGUF"
mem_ok
$PY scripts/run_exam.py --exam-dir "$EXAM_DIR" --label "$TAG-base" --parallel 1 --descriptions "$DESC" \
  || die "base run"
say "base done"

# 3. our system, short items (restart with the adapter only if there is one)
ITEM_ARGS=(--exam-dir "$EXAM_DIR" --label "$TAG-items" --parallel 1 --descriptions "$DESC" --style rag
           --rag-index data/rag/index --rag-min-score 50 --essay-mode plan --vote 9 --repair-labels --essay-retry)
if [[ -n "$EXAM_LORA" ]]; then
  start_server bielik 8080 --model "$BIELIK_GGUF" --lora "$EXAM_LORA"
  mem_ok
  ITEM_ARGS+=(--lora-scale 1)
fi
$PY scripts/run_exam.py "${ITEM_ARGS[@]}" || die "items run"
stop_server bielik
say "items done"

# 4. essay with Gemma (thinking on; 12k context keeps it under the cap)
if [[ -n "$ESSAY_IDS" && "$ESSAY_MODEL" == gemma ]]; then
  start_server gemma 8086 --model "$GEMMA_GGUF" --ctx 12288 --alias gemma
  mem_ok
  $PY scripts/run_exam.py --exam-dir "$EXAM_DIR" --label "$TAG-essay" --only "$ESSAY_IDS" --model gemma \
    --base-url http://127.0.0.1:8086/v1 --parallel 1 --descriptions "$DESC" --max-tokens-essay 10240 \
    || die "essay run"
  mem_ok
  stop_server gemma
  say "essay done"
  $PY scripts/merge_answers.py --base "runs/$TAG-items/answers.json" --override "runs/$TAG-essay/answers.json" \
    --ids "$ESSAY_IDS" --out "runs/$TAG-tuned/answers.json" || die "merge"
else
  mkdir -p "runs/$TAG-tuned" && cp "runs/$TAG-items/answers.json" "runs/$TAG-tuned/answers.json"
fi

# 5. validate both upload files
$PY scripts/validate_answers.py "runs/$TAG-base/answers.json" --exam-dir "$EXAM_DIR" || die "base file invalid"
$PY scripts/validate_answers.py "runs/$TAG-tuned/answers.json" --exam-dir "$EXAM_DIR" || die "tuned file invalid"
say "DONE in $(( $(date +%s) - T0 ))s"
echo "upload base : runs/$TAG-base/answers.json"
echo "upload tuned: runs/$TAG-tuned/answers.json"
echo "models used : speakleash/Bielik-11B-v3.0-Instruct (bartowski Q4_K_S GGUF)${EXAM_LORA:+ + LoRA $EXAM_LORA},"
[[ "$ESSAY_MODEL" == gemma ]] && echo "              google/gemma-4-12b-it (unsloth IQ4_XS GGUF, essay only),"
echo "              Qwen3.5-4B (unsloth Q8_0, image descriptions only)"
