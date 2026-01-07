#!/bin/bash
# SFT LoRA 评测启动脚本

set -euo pipefail

REPO_ROOT="/root/biyesheji"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "用法：bash project/start_sh/run_sft_eval.sh <adapter_dir> [eval_size] [max_new_tokens]" >&2
  echo "示例：bash project/start_sh/run_sft_eval.sh project/models/sft/sft_20260105_1453 200 256" >&2
  exit 1
fi

# HuggingFace：尽量和训练脚本一致，避免默认访问 huggingface.co 超时
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

ADAPTER_DIR="$1"
EVAL_SIZE="${2:-200}"
MAX_NEW_TOKENS="${3:-256}"

python -u project/src/sft/eval_sft_lora.py \
  --adapter_dir "${ADAPTER_DIR}" \
  --data_path project/data/processed/sft_data.jsonl \
  --eval_size "${EVAL_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --num_print 5


