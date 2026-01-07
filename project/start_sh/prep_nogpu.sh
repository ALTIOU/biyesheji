#!/bin/bash
# 无卡（CPU-only）准备脚本：
# - 预下载大模型/Tokenizer 到数据盘缓存（不加载进内存，避免 2GB OOM）
# - 统计数据集 token 长度分布（提前决定 max_length & 截断策略）
# - 用 tiny 模型做 SFT/GRPO 冒烟跑通（验证训练流程、产物路径、脚本参数）

set -euo pipefail

REPO_ROOT="/root/biyesheji"
cd "${REPO_ROOT}"

# HuggingFace：缓存放数据盘，避免占系统盘
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

# WandB：无卡准备阶段默认不打点（更稳）
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/wandb}"

mkdir -p "${HF_HOME}" "${WANDB_DIR}" /root/autodl-tmp/smoke

BIG_MODEL="${BIG_MODEL:-Qwen/Qwen3-1.7B}"
SFT_JSONL="${SFT_JSONL:-project/data/processed/sft_data.jsonl}"
RL_JSONL="${RL_JSONL:-project/data/processed/rl_prompts.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

echo "==> 1) 预下载（snapshot_download）: ${BIG_MODEL}"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("BIG_MODEL", "Qwen/Qwen3-1.7B")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True
print(f"Downloading snapshot: {repo_id}")
path = snapshot_download(repo_id=repo_id, token=token, ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
print("Snapshot cached at:", path)
PY

echo "==> 2) 长度统计（token/char）"
python -u project/src/utils/inspect_lengths.py --jsonl_path "${SFT_JSONL}" --format sft --tokenizer "${BIG_MODEL}" --max_length "${MAX_LENGTH}" --limit 0
python -u project/src/utils/inspect_lengths.py --jsonl_path "${RL_JSONL}"  --format rl  --tokenizer "${BIG_MODEL}" --max_length 512 --limit 0

echo "==> 3) SFT 冒烟（tiny 模型，不占内存；只跑几步验证流程）"
TINY_MODEL="${TINY_MODEL:-sshleifer/tiny-gpt2}"
SMOKE_SFT_DIR="/root/autodl-tmp/smoke/sft_tiny"
python -u project/src/sft/train_sft_lora.py \
  --data_path "${SFT_JSONL}" \
  --base_model "${TINY_MODEL}" \
  --output_dir "${SMOKE_SFT_DIR}" \
  --init_from_config \
  --max_length 128 \
  --max_train_samples 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_steps 5 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --lora_target_modules c_attn

echo "==> 4) GRPO 冒烟（tiny 模型 + tiny SFT adapter；只跑 2 条 prompt）"
SMOKE_RL_DIR="/root/autodl-tmp/smoke/rl_tiny"
python -u project/src/rl/train_grpo.py \
  --base_model "${TINY_MODEL}" \
  --sft_adapter_path "${SMOKE_SFT_DIR}" \
  --rl_data_path "${RL_JSONL}" \
  --output_dir "${SMOKE_RL_DIR}" \
  --init_from_config \
  --group_size 2 \
  --max_new_tokens 32 \
  --learning_rate 1e-4 \
  --max_prompts 2 \
  --report_to none \
  --device cpu

echo "✅ 无卡准备完成：HF 缓存已就绪 + 数据长度统计完成 + SFT/GRPO 冒烟跑通。"


