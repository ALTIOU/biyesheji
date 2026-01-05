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
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 设置环境变量，避免 OMP 冲突（Mac 环境需要）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# WandB 配置
WANDB_PROJECT = "sft_qwen3_lora"
wandb.init(project=WANDB_PROJECT)

# 路径配置
SFT_DATA_PATH = "project/data/processed/sft_data.jsonl"
BASE_MODEL = "Qwen/Qwen3-1.7B"

# 创建带时间戳的输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = f"project/models/sft/sft_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sft_dataset(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["instruction"] + (("\n" + item["input"]) if item["input"] else "")
            text = prompt + "\n\n" + item["output"]
            data_list.append({"text": text})
    return data_list

def tokenize(example):
    # M2 Mac 测试配置 - 使用较短的序列长度以节省内存
    result = tokenizer(example["text"], truncation=True, max_length=512)
    result["labels"] = result["input_ids"].copy()  # 添加 labels 用于计算 loss
    return result

    # 批量实验 GPU 配置（注释掉）：
    # result = tokenizer(example["text"], truncation=True, max_length=1024)  # GPU 可以处理更长序列
    # result["labels"] = result["input_ids"].copy()
    # return result

if __name__ == "__main__":
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("🔧 Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # M2 Mac 测试配置 - CPU 训练，不使用量化以确保兼容性
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        # device_map="auto",  # CPU 训练不需要
        torch_dtype="float32"  # CPU 模式使用 float32
    )

    # 批量实验 GPU 配置（注释掉）：
    # model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     trust_remote_code=True,
    #     device_map="auto",
    # )
    
    print("🔧 Preparing LoRA config...")
    # M2 Mac 测试配置 - 使用较小的 r 值以节省内存
    lora_config = LoraConfig(
        r=8,  # 减小 r 值，节省内存
        lora_alpha=16,  # 相应调整 alpha
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 批量实验 GPU 配置（注释掉）：
    # lora_config = LoraConfig(
    #     r=16,  # GPU 环境可以使用更大的 r 值
    #     lora_alpha=32,
    #     target_modules=["q_proj","v_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )

    # M2 Mac CPU 模式 - 不需要 prepare_model_for_kbit_training
    model = get_peft_model(model, lora_config)

    print("📥 Loading SFT dataset...")
    dataset_list = load_sft_dataset(SFT_DATA_PATH)

    # M2 Mac 测试配置 - 使用简单的数据格式
    print(f"   加载了 {len(dataset_list)} 条训练数据")

    # 只处理少量数据进行测试
    test_dataset_list = dataset_list[:50]  # 只用前10条进行测试
    print(f"   测试模式：使用 {len(test_dataset_list)} 条数据")

    tokenized_data = []
    for item in test_dataset_list:
        tokenized_item = tokenize(item)
        tokenized_data.append(tokenized_item)

    # 创建简单的 Dataset 类
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = SimpleDataset(tokenized_data)

    print("🚀 Starting SFT training...")
    # M2 Mac 测试配置 - 启用 WandB 记录
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # Mac 内存限制，使用小 batch
        gradient_accumulation_steps=4,  # 减少累积步数，加快训练
        num_train_epochs=1,  # 测试用，只训练 1 轮
        logging_steps=1,  # 更频繁的日志输出
        save_strategy="epoch",
        learning_rate=2e-4,
        # fp16=True,  # M2 Mac 不支持 fp16
        bf16=False,  # M2 Mac CPU 模式下禁用 bf16
        report_to="wandb",  # 启用 WandB 记录
        run_name=f"sft_{timestamp}",  # 每次运行的唯一名称
        # 明确指定 CPU 训练，避免 accelerate 设备检测问题
        no_cuda=True,
        dataloader_num_workers=0  # CPU 训练时避免多进程问题
    )

    # 批量实验 GPU 配置（注释掉）：
    # training_args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     per_device_train_batch_size=4,  # GPU 可以用更大 batch
    #     gradient_accumulation_steps=8,
    #     num_train_epochs=3,  # 正式训练用 3 轮
    #     logging_steps=20,
    #     save_strategy="epoch",
    #     learning_rate=2e-4,
    #     fp16=True,  # GPU 支持 fp16，效率更高
    #     report_to="wandb",  # GPU 环境使用 WandB 记录
    #     run_name=f"sft_gpu_{timestamp}"
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    print("💾 Saving LoRA SFT model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 获取训练结果
    final_loss = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None

    # 创建训练信息记录文件
    metadata = {
        "training_timestamp": timestamp,
        "training_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": BASE_MODEL,
        "data_path": SFT_DATA_PATH,
        "training_config": {
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "max_length": 512,  # tokenize 函数中的值
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
        },
        "final_loss": final_loss,
        "total_steps": trainer.state.global_step,
        "output_directory": OUTPUT_DIR,
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

    metadata_path = os.path.join(OUTPUT_DIR, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 记录到 WandB
    wandb.log({
        "final_loss": final_loss,
        "total_steps": trainer.state.global_step,
        "training_runtime": trainer.state.log_history[-1].get("train_runtime") if trainer.state.log_history else None,
        "training_config": metadata["training_config"]
    })

    # 添加标签和描述
    wandb.run.tags = ["sft", "qwen3", "lora", "mac_test"]
    wandb.run.notes = f"SFT training with Qwen3-1.7B on Mac M2. Final loss: {final_loss:.4f}"

    print(f"📝 Training metadata saved to: {metadata_path}")
    print(f"📊 WandB run URL: {wandb.run.url}")
    print("🎉 SFT training complete!")
