#!/bin/bash
# SFT 训练启动脚本

set -euo pipefail

# 仓库根目录（按你当前环境：/root/biyesheji）
REPO_ROOT="/root/biyesheji"

# （可选）激活 conda 环境：如果你不用 conda，可以把这段注释掉
CONDA_ENV_NAME="${CONDA_ENV_NAME:-biyesheji}"
if command -v conda >/dev/null 2>&1; then
  # 让 conda activate 在非交互 shell 可用
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda activate "${CONDA_ENV_NAME}"
  else
    echo "[WARN] conda 环境不存在：${CONDA_ENV_NAME}（已跳过 conda activate）" >&2
  fi
fi

# WandB：确保你已经执行过一次 `wandb login`，或提前 export WANDB_API_KEY=...
export WANDB_PROJECT="sft_qwen3_lora"
export WANDB_WATCH="false"
export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/wandb}"
# 网络不稳定时可改成：export WANDB_MODE=offline（结束后用 `wandb sync` 上传）
export WANDB_MODE="${WANDB_MODE:-online}"

# HuggingFace：国内/出网受限时建议走镜像；并把缓存放到数据盘避免占系统盘
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
# 如果镜像提示 429 限流，建议设置 HF_TOKEN（从 HuggingFace 账号生成），脚本会自动带上
# 不想明文出现在命令行：把 token 写到文件（权限 600），并设置 HF_TOKEN_FILE 指向它
export HF_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
if [[ -z "${HF_TOKEN}" ]]; then
  CANDIDATE_TOKEN_FILE=""
  if [[ -n "${HF_TOKEN_FILE}" && -f "${HF_TOKEN_FILE}" ]]; then
    CANDIDATE_TOKEN_FILE="${HF_TOKEN_FILE}"
  elif [[ -f "${HF_HOME}/token" ]]; then
    CANDIDATE_TOKEN_FILE="${HF_HOME}/token"
  elif [[ -f "${HOME}/.huggingface/token" ]]; then
    CANDIDATE_TOKEN_FILE="${HOME}/.huggingface/token"
  fi
  if [[ -n "${CANDIDATE_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(head -n 1 "${CANDIDATE_TOKEN_FILE}" | tr -d '\r\n' || true)"
    export HF_TOKEN
  fi
fi

# PyTorch：减少碎片导致的 OOM（可选但建议）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}"

# 单卡 A100 40G 推荐配置（LoRA + bf16）
# 目标：显存占用 ~35G（如果 OOM：优先降 BATCH_SIZE；还不行再降 MAX_LENGTH）
RUN_NAME="sft_a100_40g_$(date +%Y%m%d_%H%M%S)"

# 允许用环境变量覆盖关键超参
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"   # 0 表示使用全部数据
BATCH_SIZE="${BATCH_SIZE:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
USE_GC="${USE_GC:-1}" # 1 开启 gradient checkpointing（更稳），0 关闭（更吃显存）

GC_FLAG=""
if [[ "${USE_GC}" == "1" ]]; then
  GC_FLAG="--gradient_checkpointing"
fi

python -u project/src/sft/train_sft_lora.py \
  --data_path project/data/processed/sft_data.jsonl \
  --base_model Qwen/Qwen3-1.7B \
  --max_length "${MAX_LENGTH}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_train_epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --logging_steps "${LOGGING_STEPS}" \
  --bf16 \
  ${GC_FLAG} \
  --report_to wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --run_name "${RUN_NAME}"
