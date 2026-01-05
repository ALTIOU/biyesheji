# GRPO 强化学习训练脚本
# 目标：Qwen3-1.7B + LoRA，通过 group-based reward 优化生成文本的人类性
# 特点：
# - 不需要 value model（相比 PPO 更简单）
# - 适合 noisy detector reward（DetectGPT / GPTZero）
# - 当前版本：Mac 可跑调试版（GPU 代码已完整注释）

import os
import json
from datetime import datetime
import torch
import wandb
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 简易 reward（先跑通流程）
# 同目录导入：允许直接用 `python project/src/rl/train_grpo.py` 运行
from reward_functions import simple_reward

# =====================
# 基础配置
# =====================
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL_PATH = "project/models/sft"
RL_DATA_PATH = "project/data/processed/rl_prompts.jsonl"
OUTPUT_DIR = "project/models/rl"

WANDB_PROJECT = "rl_qwen3_lora"
RUN_NAME = "grpo_debug_mac"

GROUP_SIZE = 2        # 每个 prompt 采样 K 个回答（Mac 建议 2~4）
MAX_NEW_TOKENS = 80   # Mac 测试用，GPU 可调大
LR = 1e-5

# =====================
# 数据加载
# =====================
def load_rl_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    return prompts


# =====================
# LoRA 路径选择
# =====================
def resolve_sft_adapter_dir(path: str) -> str:
    """
    允许传：
    - 具体一次训练的目录：project/models/sft/sft_YYYYMMDD_HHMM
    - 或根目录：project/models/sft（会自动挑最新的 sft_* 子目录）
    """
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"SFT_MODEL_PATH not found: {path}")

    candidates: List[str] = []
    for name in os.listdir(path):
        sub = os.path.join(path, name)
        if not os.path.isdir(sub):
            continue
        if not name.startswith("sft_"):
            continue
        if os.path.isfile(os.path.join(sub, "adapter_config.json")):
            candidates.append(sub)

    if not candidates:
        raise FileNotFoundError(
            f"No adapter_config.json found under {path}. "
            f"请确认你 SFT 已产出 LoRA 目录（例如 sft_YYYYMMDD_HHMM/adapter_config.json）"
        )

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# =====================
# 计算 log probability（必须可反传）
# =====================
def compute_logprob_from_output_ids(model, output_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """
    计算 log p_theta(response | prompt)，其中 response 是 output_ids 里 prompt 后新增的 token。
    返回 shape=() 的标量张量（带梯度）。

    关键点：
    - 不能用 torch.no_grad()，否则 GRPO loss 反传为 0
    - 用 token 级切分 response，避免用字符串长度切分导致错位
    """
    # logits: [B, T, V]，我们用 next-token 预测，所以 shift 一位
    outputs = model(input_ids=output_ids)
    logits = outputs.logits[:, :-1, :]          # [B, T-1, V]
    labels = output_ids[:, 1:]                  # [B, T-1]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # response tokens 的 label 起点是 prompt_len-1（见注释）
    start = max(prompt_len - 1, 0)
    resp_token_log_probs = token_log_probs[:, start:]  # [B, response_len]
    return resp_token_log_probs.sum(dim=1).mean()


# =====================
# 主训练逻辑（GRPO）
# =====================
def main():
    run_name = f"{RUN_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project=WANDB_PROJECT, name=run_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Mac 调试版模型加载 =====
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    ).to(device)

    # 加载 SFT 后的 LoRA 权重
    adapter_dir = resolve_sft_adapter_dir(SFT_MODEL_PATH)
    print(f"Loading SFT LoRA from: {adapter_dir}")
    # 某些 peft 版本支持 is_trainable=True；不确定版本就用兼容写法
    try:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    prompts = load_rl_prompts(RL_DATA_PATH)[:3]  # Mac 只取 3 条 prompt
    global_step = 0

    for step, prompt in enumerate(prompts):
        responses = []
        rewards = []
        logprobs = []

        # ====== 1. Group Sampling ======
        for _ in range(GROUP_SIZE):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            prompt_len = input_ids.shape[1]

            # 采样不需要建图：用 no_grad（不要用 inference_mode）
            # 否则 output_ids 会变成 inference tensor，后续用于可反传的 logprob 计算会报错：
            # "Inference tensors cannot be saved for backward"
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # 额外保险：确保 output_ids 是普通 tensor（非 inference tensor）
            output_ids = output_ids.clone()

            # token 级切分 response，避免字符串长度错位
            response_ids = output_ids[:, prompt_len:]
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            reward = simple_reward(response_text)
            # logprob 需要可反传，所以单独用带梯度的 forward 计算
            logp = compute_logprob_from_output_ids(model, output_ids, prompt_len)

            responses.append(response_text)
            rewards.append(reward)
            logprobs.append(logp)

        rewards_t = torch.tensor(rewards, device=device)
        # GRPO advantage：中心化 +（可选）标准化，数值更稳
        advantages = rewards_t - rewards_t.mean()
        advantages = advantages / (rewards_t.std(unbiased=False) + 1e-6)

        # ====== 2. GRPO Loss ======
        # 最大化 E[adv * logp]  <=> 最小化 -adv * logp
        # 注意 adv 不需要梯度
        loss = 0.0
        for logp, adv in zip(logprobs, advantages):
            loss = loss + (-logp * adv.detach())
        loss = loss / GROUP_SIZE

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        wandb.log({
            "loss": loss.item(),
            "reward_mean": rewards_t.mean().item(),
            "reward_std": rewards_t.std().item(),
            "prompt_idx": step,
        }, step=global_step)

        print(f"Prompt {step} | Loss {loss.item():.4f} | Reward mean {rewards_t.mean():.4f}")
        global_step += 1

    # ===== 保存 LoRA 权重 =====
    run_out = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_out, exist_ok=True)
    model.save_pretrained(run_out)
    tokenizer.save_pretrained(run_out)
    print(f"Saved GRPO LoRA to: {run_out}")


if __name__ == "__main__":
    main()

"""
===========================
GPU / 云端正式实验修改说明
===========================
1. model.from_pretrained:
   - load_in_4bit=True
   - device_map="auto"

2. GROUP_SIZE: 8 ~ 16
3. prompts: 使用完整 300 条
4. reward: 替换为 DetectGPT / GPTZero / 组合奖励
5. optimizer / lr: 可适当调大
6. RUN_NAME: grpo_detectgpt_v1 / v2 / v3
"""
