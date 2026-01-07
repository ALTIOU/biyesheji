#!/bin/bash
# GRPO 训练启动脚本

set -euo pipefail

# 仓库根目录（按你当前环境：/root/biyesheji）
REPO_ROOT="/root/biyesheji"

# （可选）激活 conda 环境：如果你不用 conda，可以把这段注释掉
CONDA_ENV_NAME="${CONDA_ENV_NAME:-biyesheji}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda activate "${CONDA_ENV_NAME}"
  else
    echo "[WARN] conda 环境不存在：${CONDA_ENV_NAME}（已跳过 conda activate）" >&2
  fi
fi

# WandB：网络不稳可用 offline
export WANDB_PROJECT="${WANDB_PROJECT:-rl_qwen3_lora}"
export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"

# HuggingFace：尽量和 SFT 一致，避免默认访问 huggingface.co 超时
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

# PyTorch：减少碎片导致的 OOM（GPU 时有用）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}"

# 允许用环境变量覆盖关键超参/路径（A100 40G 生产默认值）
RUN_NAME="${RUN_NAME:-grpo_$(date +%Y%m%d_%H%M%S)}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-project/models/sft}"
RL_DATA_PATH="${RL_DATA_PATH:-project/data/processed/rl_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-project/models/rl}"
GROUP_SIZE="${GROUP_SIZE:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LR="${LR:-5e-6}"
MAX_PROMPTS="${MAX_PROMPTS:-0}" # 0 表示用全部
REPORT_TO="${REPORT_TO:-wandb}" # wandb / none
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
TOP_P="${TOP_P:-0.95}"
TEMPERATURE="${TEMPERATURE:-1.0}"

python -u project/src/rl/train_grpo.py \
  --base_model "${BASE_MODEL}" \
  --sft_adapter_path "${SFT_ADAPTER_PATH}" \
  --rl_data_path "${RL_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --group_size "${GROUP_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --learning_rate "${LR}" \
  --max_prompts "${MAX_PROMPTS}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --top_p "${TOP_P}" \
  --temperature "${TEMPERATURE}" \
  --bf16 \
  --device auto \
  --report_to "${REPORT_TO}" \
  --wandb_project "${WANDB_PROJECT}" \
  --run_name "${RUN_NAME}"

