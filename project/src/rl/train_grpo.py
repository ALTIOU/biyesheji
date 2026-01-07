"""
GRPO 强化学习训练脚本

目标：在 SFT LoRA 的基础上，用 group-based reward（GRPO）继续优化生成质量。

设计目标：
- AutoDL 上可一键运行（配合 project/start_sh/run_grpo_training.sh）
- 支持无卡环境“冒烟跑通”（tiny 模型 + 极少 prompts）
- GPU 来了后只需调参即可满载训练
"""

import os
import json
from datetime import datetime
import argparse
from pathlib import Path
from typing import List
import warnings

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

# 简易 reward（先跑通流程）
# 同目录导入：允许直接用 `python project/src/rl/train_grpo.py` 运行
try:
    from reward_functions import simple_reward  # type: ignore
except Exception:  # pragma: no cover
    # 兼容以 package 方式运行：python -m project.src.rl.train_grpo
    from project.src.rl.reward_functions import simple_reward  # type: ignore

# =====================
# 基础配置
# =====================
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_SFT_ADAPTER_PATH = "project/models/sft"
DEFAULT_RL_DATA_PATH = "project/data/processed/rl_prompts.jsonl"
DEFAULT_OUTPUT_DIR = "project/models/rl"
DEFAULT_WANDB_PROJECT = "rl_qwen3_lora"

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


def resolve_hf_token() -> str | bool:
    """
    与 SFT 脚本一致：优先环境变量 token，其次 token 文件；否则 token=True 让 HF 自己读已登录凭证。
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        return hf_token

    token_file = os.environ.get("HF_TOKEN_FILE")
    token_candidates = [
        token_file,
        (os.path.join(os.environ.get("HF_HOME", ""), "token") if os.environ.get("HF_HOME") else None),
        os.path.expanduser("~/.huggingface/token"),
        os.path.expanduser("~/.cache/huggingface/token"),
    ]
    for p in token_candidates:
        if not p:
            continue
        try:
            tok = Path(p).read_text(encoding="utf-8").splitlines()[0].strip()
            if tok:
                return tok
        except Exception:
            continue
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO 训练脚本（LoRA）")
    p.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--sft_adapter_path", type=str, default=DEFAULT_SFT_ADAPTER_PATH, help="可传具体 LoRA 目录或 project/models/sft 根目录")
    p.add_argument("--rl_data_path", type=str, default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    p.add_argument("--group_size", type=int, default=8, help="每个 prompt 采样 K 个回答")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_prompts", type=int, default=0, help=">0 时仅取前 N 条 prompt（适合冒烟/小跑）")

    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    p.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="auto 优先 cuda")
    p.add_argument("--bf16", action="store_true", help="CUDA 上使用 bf16（推荐 A100）")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--init_from_config",
        action="store_true",
        help="仅从 config 随机初始化模型（不加载权重）。用于无卡/低内存/torch.load 受限环境冒烟跑通。",
    )
    return p


# =====================
# 主训练逻辑（GRPO）
# =====================
def main():
    args = build_arg_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = args.run_name or f"grpo_{timestamp}"

    # WandB：需要时才初始化，避免无卡/离线环境直接报错
    wandb_run = None
    if args.report_to == "wandb":
        import wandb  # noqa: PLC0415

        wandb_run = wandb.init(project=args.wandb_project, name=run_name)

    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device != "cuda" and args.bf16:
        warnings.warn("非 CUDA 设备不支持 bf16：已忽略 --bf16。", stacklevel=2)
        args.bf16 = False

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    token_arg = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, token=token_arg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.init_from_config:
        cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, token=token_arg)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            token=token_arg,
            torch_dtype=(torch.bfloat16 if (device == "cuda" and args.bf16) else None),
        ).to(device)
    model.config.use_cache = False

    # 加载 SFT 后的 LoRA 权重
    adapter_dir = resolve_sft_adapter_dir(args.sft_adapter_path)
    print(f"Loading SFT LoRA from: {adapter_dir}")
    # 某些 peft 版本支持 is_trainable=True；不确定版本就用兼容写法
    try:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    prompts = load_rl_prompts(args.rl_data_path)
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("rl_data_path 里没有任何 prompt")
    global_step = 0

    for step, prompt in enumerate(prompts):
        # 先采样拿到 rewards，再逐个计算 logprob 并立刻反传（避免同时保留 K 个计算图导致显存飙升）
        rewards = []
        cached = []  # [(output_ids, prompt_len)]

        # ====== 1. Group Sampling ======
        for _ in range(args.group_size):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_length)
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
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # 额外保险：确保 output_ids 是普通 tensor（非 inference tensor）
            output_ids = output_ids.clone()

            # token 级切分 response，避免字符串长度错位
            response_ids = output_ids[:, prompt_len:]
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            reward = simple_reward(response_text)
            rewards.append(reward)
            cached.append((output_ids, prompt_len))

        rewards_t = torch.tensor(rewards, device=device)
        # GRPO advantage：中心化 +（可选）标准化，数值更稳
        advantages = rewards_t - rewards_t.mean()
        advantages = advantages / (rewards_t.std(unbiased=False) + 1e-6)

        # ====== 2. GRPO Loss ======
        # 最大化 E[adv * logp]  <=> 最小化 -adv * logp
        # 注意 adv 不需要梯度
        optimizer.zero_grad()
        loss_val = 0.0
        for (output_ids, prompt_len), adv in zip(cached, advantages):
            adv_detached = adv.detach()
            if device == "cuda" and args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logp = compute_logprob_from_output_ids(model, output_ids, prompt_len)
            else:
                logp = compute_logprob_from_output_ids(model, output_ids, prompt_len)
            # 逐个反传：避免累积 K 个 graph
            sample_loss = (-logp * adv_detached) / args.group_size
            loss_val += float(sample_loss.detach().item())
            sample_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if wandb_run is not None:
            import wandb  # noqa: PLC0415

            wandb.log({
                "loss": loss_val,
                "reward_mean": rewards_t.mean().item(),
                "reward_std": rewards_t.std().item(),
                "prompt_idx": step,
            }, step=global_step)

        print(f"Prompt {step} | Loss {loss_val:.4f} | Reward mean {rewards_t.mean():.4f}")
        global_step += 1

    # ===== 保存 LoRA 权重 =====
    run_out = os.path.join(args.output_dir, run_name)
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
