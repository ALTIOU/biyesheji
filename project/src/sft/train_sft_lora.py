#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) 训练脚本

目录结构：
- 每次训练会在 models/sft/ 下创建带时间戳的子目录：sft_YYYYMMDD_HHMMSS/
- 包含完整的模型文件和训练元数据

环境说明：
- M2 Mac 测试环境：当前激活配置，优化内存使用
- GPU 批量实验环境：注释掉的配置，更高效率

运行方式：
1. M2 Mac 测试（已启用 WandB）：
   cd project/mac_test
   conda activate biyesheji
   python ../src/sft/train_sft_lora.py

2. GPU 批量实验：
   - 取消注释 GPU 配置部分
   - 注释掉 Mac 配置部分
   - 调整训练参数（epochs, batch_size等）
   - 确保 WandB 已登录

WandB 项目：sft_qwen3_lora
"""

import os
import json
from datetime import datetime
import argparse
from pathlib import Path
import warnings

import torch

from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# 设置环境变量，避免 OMP 冲突（Mac 环境需要）
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WANDB_PROJECT = "sft_qwen3_lora"

def load_sft_dataset(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["instruction"] + (("\n" + item["input"]) if item["input"] else "")
            text = prompt + "\n\n" + item["output"]
            data_list.append({"text": text})
    return data_list

def build_arg_parser():
    parser = argparse.ArgumentParser(description="SFT (LoRA) 训练脚本")
    parser.add_argument("--data_path", type=str, default="project/data/processed/sft_data.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-7B")
    parser.add_argument("--output_dir", type=str, default=None, help="默认使用 project/models/sft/sft_<timestamp>")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=300, help="测试用：限制训练样本数；<=0 表示不限制")
    parser.add_argument("--max_steps", type=int, default=0, help=">0 时覆盖 epochs，按 step 训练（适合冒烟/小跑）")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="开启以显著降低显存占用（推荐 GPU）")

    parser.add_argument("--bf16", action="store_true", help="A100 推荐 bf16（默认开启）")
    parser.add_argument("--fp16", action="store_true", help="如果不支持 bf16，可用 fp16")

    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--run_name", type=str, default=None)

    # LoRA 超参（为了支持 tiny 模型冒烟跑通，以及后续快速调参）
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="逗号分隔。例如 Qwen: q_proj,v_proj；GPT2: c_attn",
    )
    parser.add_argument(
        "--init_from_config",
        action="store_true",
        help="仅从 config 随机初始化模型（不加载权重）。用于无卡/低内存/torch.load 受限环境冒烟跑通。",
    )
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # ========= 设备能力推断：无 CUDA 时不要默认开启 bf16/fp16（无卡/小内存环境很容易直接报错） =========
    has_cuda = torch.cuda.is_available()
    if not args.bf16 and not args.fp16:
        # 仅在 CUDA 可用时默认启用 bf16（A100 等）
        args.bf16 = bool(has_cuda)
    if not has_cuda and (args.bf16 or args.fp16):
        warnings.warn("检测到无 CUDA（无卡）环境：已自动关闭 bf16/fp16 以避免运行时报错。", stacklevel=2)
        args.bf16 = False
        args.fp16 = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = args.output_dir or f"project/models/sft/sft_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # WandB：只在需要时初始化，避免集群未登录/离线直接报错
    wandb_run = None
    if args.report_to == "wandb":
        import wandb  # noqa: PLC0415

        wandb_run = wandb.init(project=args.wandb_project, name=(args.run_name or f"sft_{timestamp}"))

    print(f"📁 输出目录: {output_dir}")
    print("🔧 Loading tokenizer & model...")
    # HF 鉴权：镜像/Hub 限流或私有模型时需要 token；支持 HF_TOKEN/HUGGINGFACE_HUB_TOKEN
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
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
                    hf_token = tok
                    break
            except Exception:
                continue
    # 若没有显式 token，则用 token=True 尝试读取本机已登录凭证（huggingface-cli login）
    token_arg = hf_token if hf_token else True

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=token_arg,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 训练（含 DDP/torchrun）不要用 device_map="auto"，让 Trainer/Accelerate 接管设备放置
    if args.init_from_config:
        cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, token=token_arg)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            token=token_arg,
        )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    print("🔧 Preparing LoRA config...")
    target_modules = [m.strip() for m in (args.lora_target_modules or "").split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules 不能为空（至少指定一个模块名）")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print("📥 Loading SFT dataset...")
    dataset_list = load_sft_dataset(args.data_path)
    print(f"   加载了 {len(dataset_list)} 条训练数据")

    if args.max_train_samples and args.max_train_samples > 0:
        dataset_list = dataset_list[: args.max_train_samples]
        print(f"   测试模式：使用 {len(dataset_list)} 条数据")

    train_dataset = Dataset.from_list(dataset_list)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # 交给 data collator 动态 padding，避免 batch stack 报错
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("🚀 Starting SFT training...")
    if args.bf16 and args.fp16:
        raise ValueError("bf16 和 fp16 不能同时开启，请二选一")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=("wandb" if args.report_to == "wandb" else []),
        run_name=(args.run_name or f"sft_{timestamp}"),
        max_steps=(args.max_steps if args.max_steps and args.max_steps > 0 else -1),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("💾 Saving LoRA SFT model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 获取训练结果
    final_loss = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None

    # 创建训练信息记录文件
    metadata = {
        "training_timestamp": timestamp,
        "training_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": args.base_model,
        "data_path": args.data_path,
        "training_config": {
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "max_length": args.max_length,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
        },
        "final_loss": final_loss,
        "total_steps": trainer.state.global_step,
        "output_directory": output_dir,
        "files_saved": [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
            "training_metadata.json"
        ]
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if wandb_run is not None:
        import wandb  # noqa: PLC0415

        wandb.log({
            "final_loss": final_loss,
            "total_steps": trainer.state.global_step,
            "training_runtime": trainer.state.log_history[-1].get("train_runtime") if trainer.state.log_history else None,
            "training_config": metadata["training_config"],
        })

        wandb.run.tags = ["sft", "qwen3", "lora"]
        loss_note = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "N/A"
        wandb.run.notes = f"SFT training with {args.base_model}. Final loss: {loss_note}"

    print(f"📝 Training metadata saved to: {metadata_path}")
    if wandb_run is not None:
        print(f"📊 WandB run URL: {wandb.run.url}")
    print("🎉 SFT training complete!")
