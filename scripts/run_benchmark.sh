#!/usr/bin/env bash
# Run independent PoT question workers with the release defaults.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
exec "${PYTHON:-python}" -u -m pot.parallel \
  --gpus 0 --workers-per-gpu 1 --minimum-free-mib 28000 \
  --memory-utilization 0.20 --kv-cache-gib 2.5 \
  --eval-workers 16 --eval-global-workers 64 --timeout 10 --memory-mb 1024 \
  --simulations 20 --width 3 --max-depth 8 \
  --max-new-tokens 2048 --max-context-tokens 16384 --temperature 0.7 --c-puct 1.414 \
  --learning-rate 0.0001 --epochs 3 --clip-ratio 0.3 \
  --loss-variant paper --loss-scope trajectory --beta 0.02 --base-kl-coefficient 0 \
  --reference-sync-steps 10 --lora-rank 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules q_proj k_proj v_proj o_proj \
  --disable-policy-dropout --gradient-checkpointing --dtype bfloat16 \
  --feedback public --train --greedy-commitment --seed 42 "$@"
