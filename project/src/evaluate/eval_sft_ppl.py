#!/usr/bin/env python3
"""
SFT 模型 PPL/Loss 评估脚本

用法示例：
  # Base 模型
  python project/src/evaluate/eval_sft_ppl.py \
    --base_model Qwen/Qwen3-7B \
    --data_path project/data/processed/sft_data.jsonl \
    --output_file project/results/sft_ppl_base.json \
    --num_samples 1000

  # SFT LoRA 模型
  python project/src/evaluate/eval_sft_ppl.py \
    --base_model Qwen/Qwen3-7B \
    --adapter_path project/models/sft/sft_YYYYMMDD_HHMM \
    --data_path project/data/processed/sft_data.jsonl \
    --output_file project/results/sft_ppl_sft.json \
    --num_samples 1000
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sft_dataset(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["instruction"] + (("\n" + item["input"]) if item.get("input") else "")
            output = item["output"]
            text = prompt + "\n\n" + output
            data_list.append({"prompt": prompt, "output": output, "text": text})
    return data_list


def build_arg_parser():
    parser = argparse.ArgumentParser(description="评估 SFT 模型 PPL/Loss")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-7B")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter 路径（为空则评估 base）")
    parser.add_argument("--data_path", type=str, default="project/data/processed/sft_data.jsonl")
    parser.add_argument("--output_file", type=str, default="sft_ppl_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=1000, help="评估样本数，0 表示全部")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--loss_on_output_only", action="store_true", help="仅计算输出部分的 loss/PPL（推荐）")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，但 device=cuda。请检查显卡或改为 --device cpu")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_arg = hf_token if hf_token else True

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, token=token_arg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=dtype, token=token_arg
    ).to(args.device)

    if args.adapter_path:
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    print(f"Loading data from: {args.data_path}")
    data = load_sft_dataset(args.data_path)
    if args.num_samples and args.num_samples > 0:
        data = data[: args.num_samples]
    print(f"Total samples: {len(data)}")

    total_loss = 0.0
    total_tokens = 0

    for item in tqdm(data, desc="Computing loss"):
        if args.loss_on_output_only:
            prompt = item["prompt"]
            output = item["output"]
            full_text = item["text"]

            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=args.max_length)

            input_ids = full["input_ids"].to(args.device)
            labels = input_ids.clone()

            prompt_len = min(prompt_ids["input_ids"].shape[1], labels.shape[1])
            labels[:, :prompt_len] = -100
        else:
            full = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=args.max_length)
            input_ids = full["input_ids"].to(args.device)
            labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        # 仅统计有效 token（labels != -100）
        if args.loss_on_output_only:
            valid_tokens = (labels != -100).sum().item()
        else:
            valid_tokens = labels.numel()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = float(np.exp(avg_loss))

    results = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path or "",
        "data_path": args.data_path,
        "num_samples": len(data),
        "loss_on_output_only": bool(args.loss_on_output_only),
        "avg_loss": float(avg_loss),
        "perplexity": perplexity,
    }

    Path(os.path.dirname(args.output_file) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nResults:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
