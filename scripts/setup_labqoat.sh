#!/usr/bin/env bash
# Idempotent bootstrap of a Linux GPU box (Forgehand by Labqoat, class gpu-l4: NVIDIA L4 24 GB) for training and
# serving. Every step is skipped when it is already done, so re-running after a session restart is cheap.
#
#   git clone <repo url> /workspace/fine-tune && cd /workspace/fine-tune && bash scripts/setup_labqoat.sh
#   bash scripts/setup_labqoat.sh --with-vlm                 # + image-description VLM (Qwen3.5-4B Q8_0 + F16 mmproj, docs/VISION.md)
#   bash scripts/setup_labqoat.sh --skip-build               # Python + models only
#   REPO_URL=<git url> bash setup_labqoat.sh                  # script copied alone: clones into $WORK/fine-tune first
#
# What it does (docs/RUNBOOK.md, section 1):
#   1. OS packages (git, cmake, g++, tmux, python3-venv, ...) when missing (apt-get, as root)
#   2. repo: git pull --ff-only when the checkout is clean (clone when REPO_URL is set and there is no checkout)
#   3. .venv with Python >= 3.12 + pip install -r requirements.txt (re-run only when requirements.txt changes)
#   4. Hugging Face auth check + reminder to accept the Bielik licence (the HF base repo is gated)
#   5. downloads: HF base (for training, ~9.5 GB, into the HF cache, linked as $MODELS_DIR/Bielik-4.5B-v3.0-Instruct)
#      and the official GGUF Bielik-4.5B-v3.0-Instruct.Q8_0.gguf (5.06 GB, sha256 verified) for serving
#   6. upstream llama.cpp with CUDA (llama-server, llama-tokenize) + a separate venv for convert_lora_to_gguf.py
#   7. optional: QVAC Fabric (llama.cpp fork, Vulkan, TurboQuant KV) and the qvac CLI
#   8. prints versions
#
# Flags: --skip-venv --skip-models --skip-build --with-vlm --with-qvac-fabric --with-qvac-cli --with-llama-cpp-python
#        --llama-ref REF (git ref of llama.cpp, default: current master) --cuda-arch N (default 89 = L4)
# Env:   WORK (default /workspace if it exists, else $HOME), MODELS_DIR ($WORK/models), LLAMA_CPP_DIR ($WORK/llama.cpp),
#        FABRIC_DIR ($WORK/qvac-fabric-llm.cpp), HF_TOKEN (optional; set it as a Forgehand secret)
set -euo pipefail

SKIP_VENV=0 SKIP_MODELS=0 SKIP_BUILD=0 WITH_VLM=0 WITH_FABRIC=0 WITH_QVAC_CLI=0 WITH_LCP=0
LLAMA_REF="${LLAMA_REF:-}" CUDA_ARCH="${CUDA_ARCH:-89}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-venv) SKIP_VENV=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --with-vlm) WITH_VLM=1 ;;
    --with-qvac-fabric) WITH_FABRIC=1 ;;
    --with-qvac-cli) WITH_QVAC_CLI=1 ;;
    --with-llama-cpp-python) WITH_LCP=1 ;;
    --llama-ref) LLAMA_REF="$2"; shift ;;
    --cuda-arch) CUDA_ARCH="$2"; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
  shift
done

if [[ -z "${WORK:-}" ]]; then if [[ -d /workspace ]]; then WORK=/workspace; else WORK="$HOME"; fi; fi
MODELS_DIR="${MODELS_DIR:-$WORK/models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORK/llama.cpp}"
FABRIC_DIR="${FABRIC_DIR:-$WORK/qvac-fabric-llm.cpp}"
HF_BASE_ID="speakleash/Bielik-4.5B-v3.0-Instruct"
GGUF_REPO="speakleash/Bielik-4.5B-v3.0-Instruct-GGUF"
GGUF_FILE="Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"
GGUF_SHA256="562f2291de257890adf2b4a914da8b194affe6a7a838a6b7ef3d342f306c1b7f"
GGUF_BYTES=5061215424
VLM_REPO="unsloth/Qwen3.5-4B-GGUF"                       # apache-2.0, not gated
VLM_FILES=(Qwen3.5-4B-Q8_0.gguf mmproj-F16.gguf)         # 4.48 GB + 0.67 GB
VLM_SUBDIR="vlm/qwen3.5-4b"                              # serve_model.sh --vlm looks here
NPROC="$(nproc 2>/dev/null || echo 4)"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { echo "  - $*"; }
warn() { echo "  ! $*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
# hf_get REPO            -> snapshot of the whole repo in the HF cache; prints its path
# hf_get REPO FILE [DIR] -> one file into DIR (default $MODELS_DIR; resumes partial downloads); prints its path
hf_get() {
  "$VENV/bin/python" - "$MODELS_DIR" "$@" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
models_dir, repo, *files = sys.argv[1:]
if files:
    print(hf_hub_download(repo, files[0], local_dir=files[1] if len(files) > 1 else models_dir))
else:
    print(snapshot_download(repo))
PYEOF
}

[[ "$(uname -s)" == Linux ]] || warn "this script targets the Linux GPU box; on macOS use it only as a reference"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------- 1. OS packages
step "1. OS packages"
NEED=()
for pair in git:git cmake:cmake g++:build-essential curl:curl tmux:tmux pkg-config:pkg-config; do
  have "${pair%%:*}" || NEED+=("${pair#*:}")
done
python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || NEED+=(python3-venv)
if [[ ${#NEED[@]} -gt 0 ]]; then
  if have apt-get && [[ "$(id -u)" == 0 ]]; then
    note "installing: ${NEED[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${NEED[@]}"
  else
    warn "missing packages: ${NEED[*]} (install them, then re-run)"
  fi
else
  note "all present"
fi

# ---------------------------------------------------------------- 2. repo
step "2. repository"
SCRIPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || true)"
if [[ -n "$SCRIPT_REPO" && -f "$SCRIPT_REPO/requirements.txt" ]]; then
  REPO_DIR="$SCRIPT_REPO"
elif [[ -n "${REPO_URL:-}" ]]; then
  REPO_DIR="$WORK/fine-tune"
  [[ -d "$REPO_DIR/.git" ]] || git clone "$REPO_URL" "$REPO_DIR"
else
  echo "run this from a checkout of the repo, or set REPO_URL=<git url>" >&2
  exit 1
fi
if [[ -d "$REPO_DIR/.git" ]]; then
  if [[ -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]] && git -C "$REPO_DIR" symbolic-ref -q HEAD >/dev/null; then
    git -C "$REPO_DIR" pull --ff-only || warn "git pull failed (offline or diverged); continuing with the local checkout"
  else
    note "local changes or detached HEAD: not pulling"
  fi
  note "repo $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
else
  note "repo $REPO_DIR (not a git checkout)"
fi
VENV="$REPO_DIR/.venv"

# ---------------------------------------------------------------- 3. python venv
if [[ "$SKIP_VENV" == 0 ]]; then
  step "3. Python venv (>= 3.12) + requirements"
  PYBIN=""
  for cand in python3.12 python3.13 python3; do
    if have "$cand" && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then PYBIN="$(command -v "$cand")"; break; fi
  done
  if [[ ! -x "$VENV/bin/python" ]]; then
    if [[ -n "$PYBIN" ]]; then
      "$PYBIN" -m venv "$VENV"
    else
      note "no Python >= 3.12 on PATH: using uv to fetch one"
      have uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
      uv venv --python 3.12 --seed "$VENV"
    fi
  fi
  REQ_SHA="$(sha256sum "$REPO_DIR/requirements.txt" | cut -d' ' -f1)"
  if [[ "$(cat "$VENV/.requirements.sha256" 2>/dev/null || true)" != "$REQ_SHA" ]]; then
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"
    echo "$REQ_SHA" > "$VENV/.requirements.sha256"
  else
    note "requirements already installed"
  fi
  if [[ "$WITH_LCP" == 1 ]] && ! "$VENV/bin/python" -c 'import llama_cpp' 2>/dev/null; then
    note "llama-cpp-python (CPU build; only used by check_tokenizer.py for vocab-only tokenization)"
    "$VENV/bin/python" -m pip install llama-cpp-python
  fi
  "$VENV/bin/python" -c 'import torch; print("  - torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
fi
HF="$VENV/bin/hf"; have "$HF" || HF="$(command -v hf || true)"

# ---------------------------------------------------------------- 4. Hugging Face auth
step "4. Hugging Face auth"
HF_OK=0
if [[ -n "$HF" ]] && "$HF" auth whoami >/dev/null 2>&1; then
  note "logged in as: $("$HF" auth whoami 2>/dev/null | head -n1)"
  HF_OK=1
elif [[ -n "${HF_TOKEN:-}" ]]; then
  note "HF_TOKEN is set"
  HF_OK=1
else
  warn "not logged in: run '$HF auth login' (or add HF_TOKEN as a Forgehand secret and restart the session)"
fi
GATED_OK=0
if [[ "$HF_OK" == 1 && -x "$VENV/bin/python" ]]; then
  if "$VENV/bin/python" -c "from huggingface_hub import hf_hub_download; hf_hub_download('$HF_BASE_ID', 'config.json')" >/dev/null 2>&1; then
    note "access to the gated $HF_BASE_ID: OK"
    GATED_OK=1
  else
    warn "no access to $HF_BASE_ID: accept the licence at https://huggingface.co/$HF_BASE_ID with the same HF account"
  fi
fi

# ---------------------------------------------------------------- 5. models
if [[ "$SKIP_MODELS" == 0 ]]; then
  step "5. models -> $MODELS_DIR"
  [[ -x "$VENV/bin/python" ]] || { warn "no venv (huggingface_hub): skipping downloads"; SKIP_MODELS=1; }
fi
if [[ "$SKIP_MODELS" == 0 ]]; then
  if [[ "$GATED_OK" == 1 ]]; then
    if [[ -f "$MODELS_DIR/Bielik-4.5B-v3.0-Instruct/config.json" ]] && ls "$MODELS_DIR/Bielik-4.5B-v3.0-Instruct/"*.safetensors >/dev/null 2>&1; then
      note "HF base already present"
    else
      note "downloading $HF_BASE_ID (~9.5 GB) into the HF cache"
      SNAP="$(hf_get "$HF_BASE_ID")"
      [[ -n "$SNAP" && -f "$SNAP/config.json" ]] || { echo "HF base download failed" >&2; exit 1; }
      ln -sfn "$SNAP" "$MODELS_DIR/Bielik-4.5B-v3.0-Instruct"
      note "HF base: $MODELS_DIR/Bielik-4.5B-v3.0-Instruct -> $SNAP"
    fi
  else
    warn "skipping the HF base download (no gated access); training needs it"
  fi
  GGUF_PATH="$MODELS_DIR/$GGUF_FILE"
  if [[ ! -f "$GGUF_PATH" || "$(wc -c < "$GGUF_PATH" | tr -d ' ')" != "$GGUF_BYTES" ]]; then
    note "downloading $GGUF_REPO/$GGUF_FILE (5.06 GB)"
    hf_get "$GGUF_REPO" "$GGUF_FILE" >/dev/null
  fi
  STAMP="$MODELS_DIR/.$GGUF_FILE.sha256-ok"
  if [[ ! -f "$STAMP" || "$GGUF_PATH" -nt "$STAMP" ]]; then
    note "verifying sha256 of $GGUF_FILE (5 GB, ~20 s)"
    GOT="$(sha256sum "$GGUF_PATH" | cut -d' ' -f1)"
    if [[ "$GOT" == "$GGUF_SHA256" ]]; then touch "$STAMP"; note "sha256 OK ($GOT)"
    else warn "sha256 MISMATCH for $GGUF_PATH: got $GOT, expected $GGUF_SHA256 (delete it and re-run)"; fi
  else
    note "$GGUF_FILE present, sha256 verified earlier"
  fi
  if [[ "$WITH_VLM" == 1 ]]; then
    for f in "${VLM_FILES[@]}"; do
      [[ -f "$MODELS_DIR/$VLM_SUBDIR/$f" ]] || hf_get "$VLM_REPO" "$f" "$MODELS_DIR/$VLM_SUBDIR" >/dev/null
      note "VLM file: $MODELS_DIR/$VLM_SUBDIR/$f"
    done
  fi
fi

# ---------------------------------------------------------------- 6. llama.cpp (CUDA) + converter venv
build_llama() {  # $1 = src dir, $2 = extra cmake flags, $3 = stamp label
  local src="$1" flags="$2" label="$3" rev stamp
  rev="$(git -C "$src" rev-parse HEAD)"
  stamp="$src/build/.built-$label"
  if [[ -x "$src/build/bin/llama-server" && "$(cat "$stamp" 2>/dev/null || true)" == "$rev $flags" ]]; then
    note "$label build up to date ($(git -C "$src" rev-parse --short HEAD))"
    return 0
  fi
  # shellcheck disable=SC2086
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $flags
  cmake --build "$src/build" --config Release -j "$NPROC" --target llama-server llama-tokenize
  echo "$rev $flags" > "$stamp"
}

if [[ "$SKIP_BUILD" == 0 ]]; then
  step "6. llama.cpp (upstream, CUDA) -> $LLAMA_CPP_DIR"
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  else
    git -C "$LLAMA_CPP_DIR" fetch -q origin || warn "llama.cpp fetch failed; using the local checkout"
  fi
  if [[ -n "$LLAMA_REF" ]]; then
    git -C "$LLAMA_CPP_DIR" checkout -q "$LLAMA_REF"
  elif git -C "$LLAMA_CPP_DIR" symbolic-ref -q HEAD >/dev/null; then
    git -C "$LLAMA_CPP_DIR" pull -q --ff-only || true
  fi
  if ! have nvcc && [[ -x /usr/local/cuda/bin/nvcc ]]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
  if have nvcc; then
    note "nvcc: $(nvcc --version | tail -n1)"
    build_llama "$LLAMA_CPP_DIR" "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" cuda
  else
    warn "nvcc not found: no CUDA build. Use a CUDA *devel* image for the workspace (e.g. nvidia/cuda:12.8.1-devel-ubuntu24.04)"
    warn "or apt-get install nvidia-cuda-toolkit, then re-run. Building a CPU-only llama-server for now (slow)."
    build_llama "$LLAMA_CPP_DIR" "-DGGML_CUDA=OFF" cpu
  fi
  note "llama-server: $LLAMA_CPP_DIR/build/bin/llama-server"

  CONV_VENV="$LLAMA_CPP_DIR/.venv-convert"
  CONV_REQ="$LLAMA_CPP_DIR/requirements/requirements-convert_lora_to_gguf.txt"
  if [[ -f "$CONV_REQ" ]]; then
    CONV_SHA="$(cat "$LLAMA_CPP_DIR"/requirements/requirements-convert*.txt | sha256sum | cut -d' ' -f1)"
    if [[ "$(cat "$CONV_VENV/.req.sha256" 2>/dev/null || true)" != "$CONV_SHA" ]]; then
      note "converter venv (llama.cpp pins its own torch/transformers; kept apart from the training venv)"
      [[ -x "$CONV_VENV/bin/python" ]] || "$VENV/bin/python" -m venv "$CONV_VENV"
      (cd "$LLAMA_CPP_DIR/requirements" && "$CONV_VENV/bin/python" -m pip install -q -r requirements-convert_lora_to_gguf.txt)
      echo "$CONV_SHA" > "$CONV_VENV/.req.sha256"
    else
      note "converter venv up to date"
    fi
  else
    warn "$CONV_REQ not found; export_lora_gguf.sh will fall back to the repo venv"
  fi
fi

# ---------------------------------------------------------------- 7. optional QVAC
if [[ "$WITH_FABRIC" == 1 ]]; then
  step "7a. QVAC Fabric (llama.cpp fork, Vulkan, TurboQuant KV) -> $FABRIC_DIR"
  if [[ ! -d "$FABRIC_DIR/.git" ]]; then
    git clone --filter=blob:none https://github.com/tetherto/qvac-fabric-llm.cpp "$FABRIC_DIR"
  else
    git -C "$FABRIC_DIR" pull -q --ff-only || true
  fi
  if ! have glslc || ! pkg-config --exists vulkan 2>/dev/null; then
    if have apt-get && [[ "$(id -u)" == 0 ]]; then
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan-dev glslc vulkan-tools || \
        warn "could not install the Vulkan SDK packages (glslc is in Ubuntu 24.04; older releases need the LunarG SDK)"
    fi
  fi
  if have glslc; then
    build_llama "$FABRIC_DIR" "-DGGML_VULKAN=ON" vulkan
    note "Fabric llama-server: $FABRIC_DIR/build/bin/llama-server (serve_model.sh --llama-bin $FABRIC_DIR/build/bin --backend vulkan)"
    have vulkaninfo && { vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" | head -n 4 || warn "vulkaninfo found no device (containers often lack the NVIDIA Vulkan ICD)"; }
  else
    warn "glslc missing: Fabric not built"
  fi
fi
if [[ "$WITH_QVAC_CLI" == 1 ]]; then
  step "7b. qvac CLI (npm @qvac/cli)"
  if have npm; then
    have qvac || npm i -g @qvac/cli
    qvac --version 2>/dev/null || true
  else
    warn "npm not found: install Node.js >= 18 (https://nodejs.org/en/download), then: npm i -g @qvac/cli"
  fi
fi

# ---------------------------------------------------------------- 8. versions
step "8. versions"
[[ -d "$REPO_DIR/.git" ]] && note "repo: $(git -C "$REPO_DIR" rev-parse --short HEAD) ($(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD))"
if [[ -x "$VENV/bin/python" ]]; then
  "$VENV/bin/python" - <<'PYEOF' || true
import sys
import torch, transformers, peft, trl
print(f"  - python {sys.version.split()[0]}  torch {torch.__version__} (cuda {torch.version.cuda})  "
      f"transformers {transformers.__version__}  peft {peft.__version__}  trl {trl.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  - GPU {p.name}  {p.total_memory / 2**30:.1f} GiB  compute capability {p.major}.{p.minor}")
PYEOF
fi
have nvidia-smi && note "driver: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1)"
[[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]] && note "llama-server: $("$LLAMA_CPP_DIR/build/bin/llama-server" --version 2>&1 | grep -m1 -i version || true)"
[[ -x "$FABRIC_DIR/build/bin/llama-server" ]] && note "fabric llama-server: $("$FABRIC_DIR/build/bin/llama-server" --version 2>&1 | grep -m1 -i version || true)"
note "models: $(ls "$MODELS_DIR" 2>/dev/null | tr '\n' ' ')"
note "disk: $(df -h "$WORK" | awk 'NR == 2 { print $4 " free on " $6 }')"
echo
echo "Next (docs/RUNBOOK.md): tmux new -s srv; bash scripts/serve_model.sh   |   python scripts/check_tokenizer.py --gguf $MODELS_DIR/$GGUF_FILE --llama-bin $LLAMA_CPP_DIR/build/bin"
