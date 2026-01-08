#!/usr/bin/env python3
"""
数据集长度统计（无 GPU 可跑）

用途：
- 统计字符长度、token 长度分布（p50/p95/p99/max）
- 预估 max_length 截断比例，提前发现超长样本/异常样本

支持格式：
- sft: 读取 instruction/input/output 并拼成训练 text（与 train_sft_lora.py 一致）
- rl: 读取 prompt 字段
"""

import argparse
import json
import os
from statistics import median
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JSONL 长度统计（char/token）")
    p.add_argument("--jsonl_path", type=str, required=True)
    p.add_argument("--format", type=str, default="sft", choices=["sft", "rl"])
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--max_length", type=int, default=0, help=">0 时额外统计超过该长度的比例")
    p.add_argument("--limit", type=int, default=0, help=">0 时只统计前 N 条（加速）")
    return p


def resolve_hf_token() -> str | bool:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        return hf_token
    return True


def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return int(round(d0 + d1))


def build_text(item: Dict[str, Any], fmt: str) -> str:
    if fmt == "rl":
        return str(item.get("prompt", ""))
    # sft
    instruction = str(item.get("instruction", ""))
    inp = str(item.get("input", ""))
    output = str(item.get("output", ""))
    prompt = instruction + (("\n" + inp) if inp else "")
    return prompt + "\n\n" + output


def main() -> None:
    args = build_arg_parser().parse_args()
    token_arg = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, token=token_arg)

    char_lens: List[int] = []
    tok_lens: List[int] = []
    over_max = 0
    n = 0

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = build_text(item, args.format)
            n += 1
            char_lens.append(len(text))
            tl = len(tokenizer(text, add_special_tokens=False).input_ids)
            tok_lens.append(tl)
            if args.max_length and tl > args.max_length:
                over_max += 1

    if n == 0:
        raise SystemExit("空文件或未读取到任何样本。")

    tok_lens_sorted = sorted(tok_lens)
    char_lens_sorted = sorted(char_lens)

    def stats(vals_sorted: List[int]) -> Dict[str, int]:
        return {
            "min": vals_sorted[0],
            "p50": int(median(vals_sorted)),
            "p95": percentile(vals_sorted, 95),
            "p99": percentile(vals_sorted, 99),
            "max": vals_sorted[-1],
        }

    print(f"📄 文件: {args.jsonl_path}")
    print(f"🔢 样本数: {n}")
    print(f"🔤 char_len: {stats(char_lens_sorted)}")
    print(f"🧩 tok_len : {stats(tok_lens_sorted)} (tokenizer={args.tokenizer})")
    if args.max_length and args.max_length > 0:
        ratio = over_max / n
        print(f"✂️  超过 max_length={args.max_length} 的比例: {over_max}/{n} = {ratio:.2%}")


if __name__ == "__main__":
    main()


