#!/usr/bin/env bash
# Portable PoT inference server. No machine-specific environment is sourced.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/serve_vllm.sh --model PATH_OR_REPO [options] [-- VLLM_ARGS...]

Options:
  --model PATH_OR_REPO         Required local model directory or repository ID
  --host HOST                 Bind address (default: 127.0.0.1)
  --port PORT                 Bind port (default: 8000)
  --gpu ID | --gpus IDS        CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1; default: inherit
  --tensor-parallel-size N     GPUs used by tensor parallelism (default: 1)
  --memory-utilization FLOAT   GPU memory fraction in (0,1] (default: 0.5)
  --max-model-len N            Maximum context length (default: 32768)
  --max-lora-rank N            Maximum LoRA rank (default: 8)
  --served-model-name NAME     API base-model name (default: pot-base)
  --python EXECUTABLE         Python executable, including a path (default: python)
  -h, --help                  Show this help

Examples:
  bash scripts/serve_vllm.sh --model ./models/Qwen3-4B-Instruct-2507 --gpu 0
  bash scripts/serve_vllm.sh --model ./models/my-model --gpus 0,1 \
    --tensor-parallel-size 2 --port 8001 -- --dtype bfloat16 --enforce-eager

LoRA loading is enabled for PoT's runtime adapter updates. The default address
is local-only; expose the service only within a trusted, isolated environment.
Hub requests use hf-mirror.com. Use a fully downloaded local model for offline use.
EOF
}

fail() { printf 'serve_vllm.sh: %s\n' "$1" >&2; exit 2; }
require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
}

model=''
host='127.0.0.1'
port='8000'
gpus=''
tensor_parallel='1'
memory_utilization='0.5'
max_model_len='32768'
max_lora_rank='8'
served_model_name='pot-base'
python_bin='python'
extra_args=()

while (($#)); do
  case "$1" in
    --model) require_value "$@"; model=$2; shift 2 ;;
    --host) require_value "$@"; host=$2; shift 2 ;;
    --port) require_value "$@"; port=$2; shift 2 ;;
    --gpu|--gpus) require_value "$@"; gpus=$2; shift 2 ;;
    --tensor-parallel-size) require_value "$@"; tensor_parallel=$2; shift 2 ;;
    --memory-utilization) require_value "$@"; memory_utilization=$2; shift 2 ;;
    --max-model-len) require_value "$@"; max_model_len=$2; shift 2 ;;
    --max-lora-rank) require_value "$@"; max_lora_rank=$2; shift 2 ;;
    --served-model-name) require_value "$@"; served_model_name=$2; shift 2 ;;
    --python) require_value "$@"; python_bin=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) fail "unknown option: $1 (put additional vLLM options after --)" ;;
  esac
done

[[ -n "$model" ]] || fail '--model is required'
[[ "$port" =~ ^[0-9]{1,5}$ ]] || fail '--port must be an integer in 1..65535'
((10#$port >= 1 && 10#$port <= 65535)) || fail '--port must be in 1..65535'
[[ "$memory_utilization" =~ ^0\.[0-9]*[1-9][0-9]*$ || "$memory_utilization" =~ ^1(\.0+)?$ ]] || fail '--memory-utilization must be in (0,1]'
for item in "$tensor_parallel" "$max_model_len" "$max_lora_rank"; do
  [[ "$item" =~ ^[1-9][0-9]{0,8}$ ]] || fail 'parallel size, context length and LoRA rank must be positive integers'
done
if [[ -n "$gpus" ]]; then
  [[ "$gpus" =~ ^[[:alnum:]_./-]+(,[[:alnum:]_./-]+)*$ ]] || fail '--gpu/--gpus must be comma-separated device IDs'
  export CUDA_VISIBLE_DEVICES="$gpus"
fi
command -v "$python_bin" >/dev/null 2>&1 || fail 'Python executable was not found'

export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

command=("$python_bin" -m vllm.entrypoints.openai.api_server
  --model "$model" --served-model-name "$served_model_name"
  --host "$host" --port "$port"
  --tensor-parallel-size "$tensor_parallel"
  --gpu-memory-utilization "$memory_utilization"
  --max-model-len "$max_model_len" --enable-lora --max-lora-rank "$max_lora_rank")
exec "${command[@]}" "${extra_args[@]}"
